"""K-fold fine-tuning of approved backbones on Modal GPUs (folds run in parallel).

Usage (Windows: prefix with PYTHONUTF8=1):
  python -m modal run modal_jobs/train_modal.py --upload-zip "C:/Users/Admin/Downloads/rmit-hackathon-2026.zip"   # once
  python -m modal run modal_jobs/train_modal.py --model wolf --lr 2e-5 --epochs 3
  python -m modal run modal_jobs/train_modal.py --model xlmr-large --lr 1e-5 --batch-size 16 --max-len 320 --gpu L40S

Outputs land in the Volume (runs/<run>/fold<k>.npz) and locally in outputs/runs/<run>/:
oof_probs.npy, test_probs.npy (fold mean), summary.json, submission.csv (validator-checked).
"""
from __future__ import annotations

import io
import json
import os
import time
from pathlib import Path

import modal

VOLUME_NAME = "fprinciples-c1"
VOL = "/vol"
LABELS = ["BENIGN", "JAILBREAK", "PROMPT_INJECTION"]
MODELS = {  # approved backbones only (docs/CONTEXT.md §5) — same weights as the aisuko/* Kaggle copies
    "distilbert": "distilbert/distilbert-base-multilingual-cased",
    "mmbert-small": "jhu-clsp/mmBERT-small",
    "mmbert-base": "jhu-clsp/mmBERT-base",
    "xlmr-base": "FacebookAI/xlm-roberta-base",
    "xlmr-large": "FacebookAI/xlm-roberta-large",
    "rembert": "google/rembert",
    "wolf": "patronus-studio/wolf-defender-prompt-injection",
    "deberta-pi": "protectai/deberta-v3-base-prompt-injection-v2",
}

app = modal.App("fprinciples-c1-train")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.14.1",
        "transformers==5.15.1",
        "huggingface_hub",
        "sentencepiece",
        "protobuf",
        "pandas",
        "scikit-learn",
        "numpy",
    )
    .env({"HF_HUB_DISABLE_PROGRESS_BARS": "1", "TOKENIZERS_PARALLELISM": "false"})
)


@app.function(image=image, volumes={VOL: volume}, timeout=1800)
def download_model(key: str) -> str:
    from huggingface_hub import list_repo_files, snapshot_download

    target = f"{VOL}/models/{key}"
    if not any(os.path.exists(f"{target}/{w}") for w in ("model.safetensors", "pytorch_model.bin")):
        files = list_repo_files(MODELS[key])
        weights = "model.safetensors" if "model.safetensors" in files else "pytorch_model.bin"  # mmBERT ships .bin only
        snapshot_download(MODELS[key], local_dir=target,
                          allow_patterns=["*.json", weights, "*.model", "*.txt"], ignore_patterns=["onnx/*"])
        volume.commit()
    return target


@app.function(image=image, volumes={VOL: volume}, gpu="A10G", timeout=3600)
def train_fold(cfg: dict, fold: int) -> dict:
    import math
    import random

    import numpy as np
    import pandas as pd
    import torch
    import transformers
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold
    from torch.utils.data import DataLoader
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    volume.reload()
    t_start = time.time()
    seed = cfg["seed"] + max(fold, 0)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    dev = torch.device("cuda")
    l2i = {l: i for i, l in enumerate(LABELS)}

    train = pd.read_csv(f"{VOL}/data/train.csv")
    test = pd.read_csv(f"{VOL}/data/test.csv")
    texts = train.text.astype(str).tolist()
    langs = train.language.astype(str).tolist()
    y = np.array([l2i[v] for v in train.label.astype(str).tolist()])
    strat = np.array([f"{a}_{b}" for a, b in zip(train.label.astype(str).tolist(), langs)])

    if fold >= 0:  # same fold assignment for every model -> OOF predictions are blendable
        splits = list(StratifiedKFold(cfg["n_folds"], shuffle=True, random_state=cfg["split_seed"]).split(texts, strat))
        tr_idx, va_idx = splits[fold]
    else:  # full-data refit, no validation
        tr_idx, va_idx = np.arange(len(texts)), np.array([], dtype=int)

    model_dir = f"{VOL}/models/{cfg['model']}"
    tok = AutoTokenizer.from_pretrained(model_dir)
    mcfg = AutoConfig.from_pretrained(model_dir, num_labels=3, id2label=dict(enumerate(LABELS)), label2id=l2i)
    max_len = cfg["max_len"]
    max_pos = getattr(mcfg, "max_position_embeddings", None)
    if max_pos and max_len > max_pos - 2:
        max_len = max_pos - 2
    extra = {}
    if mcfg.model_type == "modernbert":
        extra["attn_implementation"] = "sdpa"
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, config=mcfg, ignore_mismatched_sizes=True, **extra).to(dev)
    if cfg.get("grad_ckpt"):
        model.gradient_checkpointing_enable()

    with_sp, without_sp = tok("hello")["input_ids"], tok("hello", add_special_tokens=False)["input_ids"]
    k = next(i for i in range(len(with_sp)) if with_sp[i:i + len(without_sp)] == without_sp)
    prefix, suffix = with_sp[:k], with_sp[k + len(without_sp):]

    def encode(batch_texts):
        budget = max_len - len(prefix) - len(suffix)
        head = int(budget * cfg["head_ratio"])
        out = []
        for ids in tok(batch_texts, add_special_tokens=False, truncation=False)["input_ids"]:
            if len(ids) > budget:
                ids = ids[:head] + ids[-(budget - head):]
            out.append(prefix + ids + suffix)
        return out

    prep = (lambda t, lg: f"[{lg}] {t}") if cfg.get("lang_prefix") else (lambda t, lg: t)
    all_ids = encode([prep(t, lg) for t, lg in zip(texts, langs)])
    te_ids = encode([prep(t, lg) for t, lg in zip(test.text.astype(str).tolist(), test.language.astype(str).tolist())])
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0

    def loader(seqs, labels=None, shuffle=False, bs=None):
        order = list(range(len(seqs)))
        if not shuffle:
            order.sort(key=lambda i: len(seqs[i]))
        items = [(i, seqs[i], -1 if labels is None else int(labels[i])) for i in order]

        def collate(batch):
            idx, ss, ys = zip(*batch)
            m = max(len(s) for s in ss)
            return (torch.tensor(idx), torch.tensor([s + [pad] * (m - len(s)) for s in ss]),
                    torch.tensor([[1] * len(s) + [0] * (m - len(s)) for s in ss]), torch.tensor(ys))
        return DataLoader(items, batch_size=bs or cfg["batch_size"], shuffle=shuffle, collate_fn=collate, num_workers=2)

    amp_dtype = torch.bfloat16 if cfg["precision"] == "bf16" else torch.float16
    use_scaler = cfg["precision"] == "fp16"

    ema = None

    @torch.no_grad()
    def predict(seqs):
        net = ema.module if ema is not None else model
        net.eval()
        probs = np.zeros((len(seqs), 3), dtype=np.float32)
        for idx, x, a, _ in loader(seqs, bs=cfg["batch_size"] * 2):
            with torch.autocast("cuda", dtype=amp_dtype):
                logits = net(input_ids=x.to(dev), attention_mask=a.to(dev)).logits
            probs[idx.numpy()] = torch.softmax(logits.float(), -1).cpu().numpy()
        return probs

    tr_seqs = [all_ids[i] for i in tr_idx]
    tr_labels = y[tr_idx]
    va_seqs = [all_ids[i] for i in va_idx]
    n_pseudo = 0
    if cfg.get("pseudo"):  # confident test predictions of an earlier ensemble as extra hard labels (train side only)
        pp = np.load(f"{VOL}/pseudo/{cfg['pseudo']}.npy")
        keep = np.where(pp.max(1) >= cfg["pseudo_thr"])[0]
        tr_seqs = tr_seqs + [te_ids[i] for i in keep]
        tr_labels = np.concatenate([tr_labels, pp[keep].argmax(1)])
        n_pseudo = len(keep)
    # per-sample weights: up-weight the hard slice (long Portuguese BENIGN/JAILBREAK prompts)
    tr_w = np.ones(len(tr_seqs), dtype=np.float32)
    if cfg.get("hard_weight", 1.0) != 1.0:
        lens = np.array([len(texts[i]) for i in tr_idx])
        hard = (np.array([langs[i] for i in tr_idx]) == "pt") & (lens > 400) & (y[tr_idx] < 2)
        tr_w[:len(tr_idx)][hard] = cfg["hard_weight"]
    tr_w_t = torch.tensor(tr_w, device=dev)
    freq = np.bincount(tr_labels, minlength=3) / len(tr_labels)
    cw = (1.0 / freq) ** cfg["class_weight_power"]
    cw = cw / cw.mean()
    ce = torch.nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float, device=dev),
                                   label_smoothing=cfg.get("label_smoothing", 0.0), reduction="none")
    cw_t = torch.tensor(cw, dtype=torch.float, device=dev)

    def loss_fn(logits, target, idx=None):
        per = ce(logits, target)
        sw = tr_w_t[idx] if idx is not None else torch.ones_like(per)
        return (per * sw).sum() / (cw_t[target] * sw).sum()  # weighted mean, same scale as the default CE
    train_loader = loader(tr_seqs, tr_labels, shuffle=True)
    steps = math.ceil(len(train_loader) / cfg["grad_accum"]) * cfg["epochs"]

    # Layer-wise LR decay (llrd=1.0 -> plain AdamW): lr_layer = lr * llrd^(depth from the top)
    import re
    no_decay = ("bias", "LayerNorm.weight", "layer_norm", "norm.weight")
    n_layers = getattr(mcfg, "num_hidden_layers", 12)

    def depth_scale(name):
        m = re.search(r"\.layers?\.(\d+)\.", name)
        if m:
            return cfg["llrd"] ** (n_layers - int(m.group(1)))
        if "embed" in name:
            return cfg["llrd"] ** (n_layers + 1)
        return 1.0  # pooler / head / final norm

    groups = [{"params": [p], "lr": cfg["lr"] * depth_scale(n),
               "weight_decay": 0.0 if any(s in n for s in no_decay) else 0.01}
              for n, p in model.named_parameters()]
    opt = torch.optim.AdamW(groups, lr=cfg["lr"])
    sched = get_linear_schedule_with_warmup(opt, int(cfg["warmup_ratio"] * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    if cfg.get("ema"):
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
        ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(cfg["ema"]), use_buffers=True)
    emb = model.get_input_embeddings().weight

    history, best = [], {"f1": -1.0}
    for ep in range(cfg["epochs"]):
        model.train()
        t0, run_loss = time.time(), 0.0
        for step, (bidx, x, a, yy) in enumerate(train_loader):
            x, a, yy, bidx = x.to(dev), a.to(dev), yy.to(dev), bidx.to(dev)
            with torch.autocast("cuda", dtype=amp_dtype):
                logits = model(input_ids=x, attention_mask=a).logits
            loss = loss_fn(logits.float(), yy, bidx) / cfg["grad_accum"]
            scaler.scale(loss).backward()
            if cfg.get("fgm") and emb.grad is not None:  # FGM: perturb only the embedding rows used in this batch
                rows = torch.unique(x)
                g = emb.grad[rows]
                norm = g.norm()
                if torch.isfinite(norm) and norm > 0:
                    delta = cfg["fgm"] * g / norm
                    emb.data[rows] += delta
                    with torch.autocast("cuda", dtype=amp_dtype):
                        adv_logits = model(input_ids=x, attention_mask=a).logits
                    scaler.scale(loss_fn(adv_logits.float(), yy, bidx) / cfg["grad_accum"]).backward()
                    emb.data[rows] -= delta
            if (step + 1) % cfg["grad_accum"] == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
                if ema is not None:
                    ema.update_parameters(model)
            run_loss += loss.item()
        rec = {"epoch": ep, "loss": run_loss / len(train_loader), "train_s": round(time.time() - t0)}
        if len(va_idx):
            vp = predict(va_seqs)
            rec["val_macro_f1"] = float(f1_score(y[va_idx], vp.argmax(1), average="macro", labels=[0, 1, 2]))
            if rec["val_macro_f1"] > best["f1"]:
                best = {"f1": rec["val_macro_f1"], "epoch": ep, "val_probs": vp, "test_probs": predict(te_ids)}
        history.append(rec)
        print(f"[{cfg['run']} fold {fold}] {rec}", flush=True)
    if not len(va_idx):
        best = {"f1": None, "epoch": cfg["epochs"] - 1, "val_probs": np.zeros((0, 3), np.float32), "test_probs": predict(te_ids)}

    out_dir = f"{VOL}/runs/{cfg['run']}"
    os.makedirs(out_dir, exist_ok=True)
    np.savez(f"{out_dir}/fold{fold}.npz", val_idx=va_idx, val_probs=best["val_probs"], test_probs=best["test_probs"])
    meta = {"fold": fold, "best_epoch": best["epoch"], "best_val_macro_f1": best["f1"], "history": history,
            "max_len_used": max_len, "n_pseudo": n_pseudo, "transformers": transformers.__version__, "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0), "minutes": round((time.time() - t_start) / 60, 1)}
    Path(f"{out_dir}/fold{fold}.json").write_text(json.dumps(meta, indent=1))
    volume.commit()
    return meta


def upload_data(zip_path: str) -> None:
    import zipfile

    z = zipfile.ZipFile(zip_path)
    with volume.batch_upload(force=True) as batch:
        for name in ("train.csv", "test.csv", "sample_submission.csv"):
            batch.put_file(io.BytesIO(z.read(name)), f"/data/{name}")
    print("uploaded data to volume", VOLUME_NAME)


def aggregate(run: str, folds: list[int], cfg: dict) -> None:
    """Pull fold results, build OOF, report metrics, fit class multipliers, write a validated submission."""
    import sys

    import numpy as np
    import pandas as pd

    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "src"))
    sys.path.insert(0, str(repo / "scripts"))
    from c1.metrics import apply_class_weights, fit_class_weights, report
    from validate_submission import validate

    out = repo / "outputs" / "runs" / run
    out.mkdir(parents=True, exist_ok=True)

    def fetch(name: str) -> bytes:
        return b"".join(volume.read_file(f"runs/{run}/{name}"))

    def fetch_data(name: str) -> pd.DataFrame:
        return pd.read_csv(io.BytesIO(b"".join(volume.read_file(f"data/{name}"))))

    train, sample = fetch_data("train.csv"), fetch_data("sample_submission.csv")
    y = np.array([LABELS.index(v) for v in train.label.astype(str).tolist()])
    langs = np.array(train.language.astype(str).tolist())
    oof = np.full((len(train), 3), np.nan, dtype=np.float32)
    tests, metas = [], []
    for k in folds:
        d = np.load(io.BytesIO(fetch(f"fold{k}.npz")))
        if len(d["val_idx"]):
            oof[d["val_idx"]] = d["val_probs"]
        tests.append(d["test_probs"])
        metas.append(json.loads(fetch(f"fold{k}.json")))
    test_probs = np.mean(tests, axis=0)
    covered = ~np.isnan(oof).any(1)
    summary = {"run": run, "cfg": cfg, "folds": metas, "oof_coverage": float(covered.mean())}
    weights = np.ones(3)
    if covered.any():
        summary["oof_raw"] = report(y[covered], oof[covered].argmax(1), langs[covered])
        weights, tuned = fit_class_weights(oof[covered], y[covered])
        summary["class_multipliers"] = weights.tolist()
        summary["oof_tuned"] = report(y[covered], apply_class_weights(oof[covered], weights), langs[covered])
        np.save(out / "oof_probs.npy", oof)
    np.save(out / "test_probs.npy", test_probs)
    pred = apply_class_weights(test_probs, weights)
    sub = pd.DataFrame({"id": sample.id.astype(str).tolist(), "label": [LABELS[i] for i in pred]})
    sub.to_csv(out / "submission.csv", index=False)
    sample.to_csv(out / "_sample.csv", index=False)
    errors = validate(out / "_sample.csv", out / "submission.csv")
    summary["submission_valid"] = not errors
    summary["submission_errors"] = errors
    summary["pred_distribution"] = sub.label.value_counts(normalize=True).round(4).to_dict()
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary.get(k) for k in ("oof_raw", "class_multipliers", "oof_tuned", "pred_distribution",
                                                  "submission_valid")}, indent=1))
    print("fold minutes:", [m["minutes"] for m in metas], "| best epochs:", [m["best_epoch"] for m in metas])
    print("local outputs:", out)


@app.local_entrypoint()
def main(model: str = "wolf", lr: float = 2e-5, epochs: int = 3, batch_size: int = 32, max_len: int = 384,
         grad_accum: int = 1, folds: str = "0,1,2,3,4", n_folds: int = 5, gpu: str = "A10G",
         precision: str = "bf16", class_weight_power: float = 0.5, label_smoothing: float = 0.0,
         head_ratio: float = 0.7, lang_prefix: bool = False, grad_ckpt: bool = False, seed: int = 42,
         llrd: float = 1.0, ema: float = 0.0, fgm: float = 0.0, pseudo: str = "", pseudo_thr: float = 0.98,
         hard_weight: float = 1.0,
         run: str = "", upload_zip: str = "", aggregate_only: bool = False):
    if upload_zip:
        upload_data(upload_zip)
        return
    if pseudo:  # upload the local test probabilities of an earlier run/blend as the pseudo-label source
        src = Path(__file__).resolve().parents[1] / "outputs" / "runs" / pseudo / "test_probs.npy"
        with volume.batch_upload(force=True) as batch:
            batch.put_file(str(src), f"/pseudo/{pseudo}.npy")
    assert model in MODELS, f"model must be one of {list(MODELS)}"
    fold_list = [int(f) for f in folds.split(",")]
    run = run or f"{model}_lr{lr:g}_ep{epochs}_len{max_len}_{time.strftime('%H%M')}"
    cfg = dict(model=model, lr=lr, epochs=epochs, batch_size=batch_size, max_len=max_len, grad_accum=grad_accum,
               n_folds=n_folds, split_seed=42, precision=precision, class_weight_power=class_weight_power,
               label_smoothing=label_smoothing, head_ratio=head_ratio, lang_prefix=lang_prefix,
               grad_ckpt=grad_ckpt, warmup_ratio=0.1, seed=seed, run=run, gpu=gpu,
               llrd=llrd, ema=ema, fgm=fgm, pseudo=pseudo, pseudo_thr=pseudo_thr, hard_weight=hard_weight)
    print("run:", run, json.dumps(cfg))
    if not aggregate_only:
        print("model dir:", download_model.remote(model))
        t0 = time.time()
        for res in train_fold.with_options(gpu=gpu).map([cfg] * len(fold_list), fold_list, return_exceptions=True):
            print("fold done:", res if isinstance(res, Exception) else
                  {k: res[k] for k in ("fold", "best_epoch", "best_val_macro_f1", "minutes", "gpu")})
        print(f"all folds finished in {(time.time() - t0) / 60:.1f} min")
    aggregate(run, fold_list, cfg)
