# %% [markdown]
# # RMIT_<CAMPUS>_Fprinciples_C1 — Multilingual prompt-attack classifier
# **Model used: `jhu-clsp/mmBERT-small`** (approved backbone, Kaggle model `aisuko/mmbert-small`) ← update when you change MODEL_SLUG
#
# Self-contained: paste into one Kaggle notebook (GPU T4/P100, Internet off is fine).
# Inputs to attach: competition data `rmit-hackathon-2026` + the chosen model from `aisuko/*`.
# Outputs in /kaggle/working: submission.csv, val_probs.npy, val_labels.npy, test_probs.npy, run.json

# %%
import glob, json, math, os, random, time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

# %% ---------------- config ----------------
CFG = dict(
    model_slug="mmbert-small",   # folder name under /kaggle/input (mmbert-small, mmbert-base, xlm-roberta-large, wolf-defender-prompt-injection, ...)
    max_len=256,                 # mean text ~294 chars ≈ 80-120 tokens; 256 covers most; long texts use head+tail
    head_ratio=0.7,              # share of tokens kept from the start when truncating
    epochs=2,
    lr=5e-5,                     # xlm-roberta-large: 1e-5; mmBERT-base / xlmr-base: 3e-5
    batch_size=32,               # xlmr-large: 16 (or 8 + grad_accum 2); rembert: 8
    grad_accum=1,
    warmup_ratio=0.1,
    weight_decay=0.01,
    class_weight_power=0.5,      # loss weight = (1/freq)^power; 0 = plain CE
    lang_prefix=False,           # prepend "[pt] " etc. as a controlled experiment
    val_size=0.15,
    seed=42,
    fp16=True,                   # T4 has no bf16
    run_name="s1_mmbert_small",
)
LABELS = ["BENIGN", "JAILBREAK", "PROMPT_INJECTION"]
L2I = {l: i for i, l in enumerate(LABELS)}
OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "outputs"
os.makedirs(OUT, exist_ok=True)
print(json.dumps(CFG, indent=1))

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_all(CFG["seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

# %% ---------------- locate data + model ----------------
def find_one(pattern):
    hits = sorted(glob.glob(pattern, recursive=True))
    assert hits, f"not found: {pattern}"
    return hits[0]

# Local runs can override with env vars DATA_DIR / MODEL_DIR
data_dir = os.environ.get("DATA_DIR") or os.path.dirname(find_one("/kaggle/input/**/train.csv"))
MODEL_DIR = os.environ.get("MODEL_DIR")
if not MODEL_DIR:
    model_cfg = [p for p in glob.glob("/kaggle/input/**/config.json", recursive=True) if CFG["model_slug"] in p.lower()]
    assert model_cfg, f"attach the Kaggle model for {CFG['model_slug']}; found: {glob.glob('/kaggle/input/*')}"
    MODEL_DIR = os.path.dirname(sorted(model_cfg, key=len)[0])
print("data:", data_dir, "\nmodel:", MODEL_DIR)

train = pd.read_csv(f"{data_dir}/train.csv")
test = pd.read_csv(f"{data_dir}/test.csv")
sample = pd.read_csv(f"{data_dir}/sample_submission.csv")
print(train.shape, test.shape, sample.shape)
print(pd.crosstab(train.language, train.label, margins=True))

# %% ---------------- quick audit ----------------
dups = train.duplicated("text").sum()
conflict = train.groupby("text").label.nunique().gt(1).sum()
overlap = test.text.isin(set(train.text)).sum()
print(f"exact duplicate texts in train: {dups} | conflicting labels: {conflict} | test texts also in train: {overlap}")
print(train.text.str.len().describe())

# %% ---------------- split (stratified by label x language) ----------------
train["y"] = train.label.map(L2I)
strat = train.label + "_" + train.language
tr_df, va_df = train_test_split(train, test_size=CFG["val_size"], stratify=strat, random_state=CFG["seed"])
# keep exact-duplicate texts on one side to avoid leakage
leak = va_df.text.isin(set(tr_df.text))
tr_df = pd.concat([tr_df, va_df[leak]]); va_df = va_df[~leak]
print("train/val:", len(tr_df), len(va_df), "| moved duplicates:", int(leak.sum()))

# %% ---------------- tokenisation (head+tail truncation) ----------------
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
_max_pos = getattr(AutoConfig.from_pretrained(MODEL_DIR), "max_position_embeddings", None)
if _max_pos and CFG["max_len"] > _max_pos - 2:  # RoBERTa-style models reserve 2 positions
    print(f"max_len {CFG['max_len']} -> {_max_pos - 2} (model limit)"); CFG["max_len"] = _max_pos - 2

def prep_text(df):
    if CFG["lang_prefix"]:
        return ("[" + df.language + "] " + df.text.astype(str)).tolist()
    return df.text.astype(str).tolist()

# Derive the special-token prefix/suffix empirically (works on transformers 4.x and 5.x, any tokenizer)
_with, _without = tok("hello")["input_ids"], tok("hello", add_special_tokens=False)["input_ids"]
_k = next(i for i in range(len(_with)) if _with[i:i + len(_without)] == _without)
PREFIX, SUFFIX = _with[:_k], _with[_k + len(_without):]
print("special prefix/suffix:", PREFIX, SUFFIX)

def encode(texts):
    budget = CFG["max_len"] - len(PREFIX) - len(SUFFIX)
    head = int(budget * CFG["head_ratio"])
    out = []
    for ids in tok(texts, add_special_tokens=False, truncation=False)["input_ids"]:
        if len(ids) > budget:
            ids = ids[:head] + ids[-(budget - head):]
        out.append(PREFIX + ids + SUFFIX)
    return out

def make_loader(ids, labels=None, shuffle=False, bs=None):
    bs = bs or CFG["batch_size"]
    order = list(range(len(ids)))
    if not shuffle:  # length-sorted inference = fast; restore order later
        order = sorted(order, key=lambda i: len(ids[i]))
    items = [(i, ids[i], -1 if labels is None else int(labels[i])) for i in order]
    def collate(batch):
        idx, seqs, ys = zip(*batch)
        m = max(len(s) for s in seqs)
        pad = tok.pad_token_id
        input_ids = torch.tensor([s + [pad] * (m - len(s)) for s in seqs])
        attn = torch.tensor([[1] * len(s) + [0] * (m - len(s)) for s in seqs])
        return torch.tensor(idx), input_ids, attn, torch.tensor(ys)
    return DataLoader(items, batch_size=bs, shuffle=shuffle, collate_fn=collate, num_workers=2)

t0 = time.time()
tr_ids, va_ids, te_ids = encode(prep_text(tr_df)), encode(prep_text(va_df)), encode(prep_text(test))
lens = np.array([len(x) for x in tr_ids])
print(f"tokenised in {time.time()-t0:.0f}s | train len p50={np.median(lens):.0f} p95={np.percentile(lens,95):.0f} "
      f"| truncated share={np.mean(lens>=CFG['max_len']):.1%}")

# %% ---------------- model ----------------
import transformers
mcfg = AutoConfig.from_pretrained(MODEL_DIR, num_labels=3, id2label=dict(enumerate(LABELS)), label2id=L2I)
extra = {}
if mcfg.model_type == "modernbert":  # mmBERT-small/base, wolf-defender
    extra["attn_implementation"] = "sdpa"  # FlashAttention-2 does not run on T4
    if hasattr(mcfg, "reference_compile"):
        mcfg.reference_compile = False
    # wolf-defender's config was saved with transformers 5.x (rope_parameters); on 4.x the
    # local RoPE theta silently falls back to 10000 — pin it to the trained value.
    if int(transformers.__version__.split(".")[0]) < 5 and getattr(mcfg, "global_rope_theta", None) == 160000.0:
        mcfg.local_rope_theta = 160000.0
    print("transformers", transformers.__version__, "| rope global/local:",
          getattr(mcfg, "global_rope_theta", None), getattr(mcfg, "local_rope_theta", None))
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR, config=mcfg, ignore_mismatched_sizes=True,  # replaces 2-class heads of safety-tuned models
    **extra,
).to(device)

freq = np.bincount(tr_df.y, minlength=3) / len(tr_df)
cw = (1.0 / freq) ** CFG["class_weight_power"]; cw = cw / cw.mean()
print("class weights:", dict(zip(LABELS, cw.round(3))))
loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float, device=device))

train_loader = make_loader(tr_ids, tr_df.y.values, shuffle=True)
steps = math.ceil(len(train_loader) / CFG["grad_accum"]) * CFG["epochs"]
no_decay = ["bias", "LayerNorm.weight", "layer_norm", "norm.weight"]
params = [
    {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)], "weight_decay": CFG["weight_decay"]},
    {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)], "weight_decay": 0.0},
]
opt = torch.optim.AdamW(params, lr=CFG["lr"])
sched = get_linear_schedule_with_warmup(opt, int(CFG["warmup_ratio"] * steps), steps)
scaler = torch.amp.GradScaler("cuda", enabled=CFG["fp16"] and device.type == "cuda")

@torch.no_grad()
def predict_probs(ids):
    model.eval()
    probs = np.zeros((len(ids), 3), dtype=np.float32)
    for idx, x, a, _ in make_loader(ids, bs=CFG["batch_size"] * 2):
        with torch.autocast(device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
            logits = model(input_ids=x.to(device), attention_mask=a.to(device)).logits
        probs[idx.numpy()] = torch.softmax(logits.float(), -1).cpu().numpy()
    return probs

def macro(y, p):
    return f1_score(y, p, average="macro", labels=[0, 1, 2])

# %% ---------------- train ----------------
best_f1, best_state = -1, None
for ep in range(CFG["epochs"]):
    model.train(); t0 = time.time(); run_loss = 0.0
    for step, (_, x, a, y) in enumerate(train_loader):
        with torch.autocast(device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
            logits = model(input_ids=x.to(device), attention_mask=a.to(device)).logits
        loss = loss_fn(logits.float(), y.to(device)) / CFG["grad_accum"]
        scaler.scale(loss).backward()
        if (step + 1) % CFG["grad_accum"] == 0:
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
        run_loss += loss.item()
        if step % 100 == 0:
            print(f"ep{ep} step {step}/{len(train_loader)} loss {run_loss/(step+1):.4f}")
    vp = predict_probs(va_ids)
    f1 = macro(va_df.y.values, vp.argmax(1))
    print(f"== epoch {ep}: val macro-F1 {f1:.4f} ({time.time()-t0:.0f}s)")
    if f1 > best_f1:
        best_f1, best_state = f1, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
model.load_state_dict(best_state)

# %% ---------------- validation report + per-class multipliers (fit on VAL only) ----------------
val_probs = predict_probs(va_ids)
y_val = va_df.y.values
print(classification_report(y_val, val_probs.argmax(1), target_names=LABELS, digits=4))
for lang in sorted(va_df.language.unique()):
    m = (va_df.language == lang).values
    print(f"  {lang}: macro-F1 {macro(y_val[m], val_probs[m].argmax(1)):.4f}  (n={m.sum()})")

w = np.ones(3); base = best = macro(y_val, val_probs.argmax(1))
for _ in range(3):
    for c in (1, 2):
        for g in np.arange(0.3, 4.01, 0.05):
            t = w.copy(); t[c] = g
            s = macro(y_val, (val_probs * t).argmax(1))
            if s > best + 1e-6: best, w = s, t
print(f"val macro-F1 raw {base:.4f} -> with class multipliers {best:.4f}  weights={w.round(2)}")
# NOTE: multipliers fitted and evaluated on the same val split are optimistic; keep them only if the gain is clear.

# %% ---------------- predict test + write submission ----------------
test_probs = predict_probs(te_ids)
pred = (test_probs * w).argmax(1)
sub = pd.DataFrame({"id": test.id, "label": [LABELS[i] for i in pred]})
sub = sample[["id"]].merge(sub, on="id", how="left")
assert len(sub) == 6000 and sub.label.isin(LABELS).all() and (sub.id.values == sample.id.values).all()
sub.to_csv(f"{OUT}/submission.csv", index=False)
np.save(f"{OUT}/val_probs.npy", val_probs); np.save(f"{OUT}/val_labels.npy", y_val)
np.save(f"{OUT}/val_ids.npy", va_df.id.values); np.save(f"{OUT}/test_probs.npy", test_probs)
json.dump({**CFG, "model_dir": MODEL_DIR, "val_macro_f1": base, "val_macro_f1_weighted": best,
           "class_multipliers": w.tolist(), "pred_dist": sub.label.value_counts().to_dict()},
          open(f"{OUT}/run.json", "w"), indent=1)
print(sub.label.value_counts(normalize=True))
print("saved to", OUT)
