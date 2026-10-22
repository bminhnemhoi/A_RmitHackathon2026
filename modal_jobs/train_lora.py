"""LoRA fine-tuning of the approved decoder LLMs (Mistral-7B-Instruct-v0.3, Granite-3.3-2B) as 3-class classifiers.

Head: last-token hidden state -> Linear(3). Same 5-fold split and output format as train_modal.py,
so runs blend directly with the encoder runs.

  python -m modal run modal_jobs/train_lora.py --model mistral --gpu H100
  python -m modal run modal_jobs/train_lora.py --model granite --gpu L40S --lr 2e-4 --epochs 2
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import modal

VOL = "/vol"
LABELS = ["BENIGN", "JAILBREAK", "PROMPT_INJECTION"]
LLMS = {  # approved backbones only
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "granite": "ibm-granite/granite-3.3-2b-instruct",
}

app = modal.App("fprinciples-c1-lora")
volume = modal.Volume.from_name("fprinciples-c1", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.14.1", "transformers==5.15.1", "peft", "accelerate", "huggingface_hub",
                 "sentencepiece", "protobuf", "pandas", "scikit-learn", "numpy")
    .env({"HF_HUB_DISABLE_PROGRESS_BARS": "1", "TOKENIZERS_PARALLELISM": "false"})
)


@app.function(image=image, volumes={VOL: volume}, timeout=3600)
def download_llm(key: str) -> str:
    from huggingface_hub import snapshot_download

    target = f"{VOL}/models/{key}"
    if not os.path.exists(f"{target}/model.safetensors.index.json"):
        snapshot_download(LLMS[key], local_dir=target,
                          allow_patterns=["*.json", "model-*.safetensors", "tokenizer*", "*.model"],
                          ignore_patterns=["consolidated*"])
        volume.commit()
    return target


@app.function(image=image, volumes={VOL: volume}, gpu="H100", timeout=5400)
def train_fold(cfg: dict, fold: int) -> dict:
    import math
    import random

    import numpy as np
    import pandas as pd
    import torch
    from peft import LoraConfig, get_peft_model
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold
    from torch.utils.data import DataLoader
    from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

    volume.reload()
    t_start = time.time()
    seed = cfg["seed"] + fold
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    dev = torch.device("cuda")
    l2i = {l: i for i, l in enumerate(LABELS)}

    train = pd.read_csv(f"{VOL}/data/train.csv")
    test = pd.read_csv(f"{VOL}/data/test.csv")
    texts = train.text.astype(str).tolist()
    langs = train.language.astype(str).tolist()
    y = np.array([l2i[v] for v in train.label.astype(str).tolist()])
    strat = np.array([f"{a}_{b}" for a, b in zip(train.label.astype(str).tolist(), langs)])
    splits = list(StratifiedKFold(cfg["n_folds"], shuffle=True, random_state=cfg["split_seed"]).split(texts, strat))
    tr_idx, va_idx = splits[fold]

    model_dir = f"{VOL}/models/{cfg['model']}"
    tok = AutoTokenizer.from_pretrained(model_dir)
    pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    eos = [tok.eos_token_id] if tok.eos_token_id is not None else []

    def encode(batch_texts):
        budget = cfg["max_len"] - len(bos) - len(eos)
        head = int(budget * 0.7)
        out = []
        for ids in tok(batch_texts, add_special_tokens=False, truncation=False)["input_ids"]:
            if len(ids) > budget:
                ids = ids[:head] + ids[-(budget - head):]
            out.append(bos + ids + eos)  # the last token (EOS) is the pooled "summary" position
        return out

    all_ids = encode(texts)
    te_ids = encode(test.text.astype(str).tolist())

    base = AutoModel.from_pretrained(model_dir, dtype=torch.bfloat16).to(dev)
    if cfg.get("grad_ckpt"):
        base.gradient_checkpointing_enable()
        base.enable_input_require_grads()
    lora = LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_r"] * 2, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    base = get_peft_model(base, lora)
    hidden = base.config.hidden_size
    head = torch.nn.Linear(hidden, 3).to(dev)  # fp32 head

    def forward(x, a):
        h = base(input_ids=x, attention_mask=a).last_hidden_state
        last = a.sum(1) - 1  # right padding -> index of the last real token
        pooled = h[torch.arange(h.size(0), device=dev), last].float()
        return head(pooled)

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

    @torch.no_grad()
    def predict(seqs):
        base.eval(); head.eval()
        probs = np.zeros((len(seqs), 3), dtype=np.float32)
        for idx, x, a, _ in loader(seqs, bs=cfg["batch_size"] * 2):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = forward(x.to(dev), a.to(dev))
            probs[idx.numpy()] = torch.softmax(logits.float(), -1).cpu().numpy()
        return probs

    tr_seqs = [all_ids[i] for i in tr_idx]
    va_seqs = [all_ids[i] for i in va_idx]
    freq = np.bincount(y[tr_idx], minlength=3) / len(tr_idx)
    cw = (1.0 / freq) ** 0.5
    cw = cw / cw.mean()
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float, device=dev))
    train_loader = loader(tr_seqs, y[tr_idx], shuffle=True)
    steps = math.ceil(len(train_loader) / cfg["grad_accum"]) * cfg["epochs"]
    opt = torch.optim.AdamW([
        {"params": [p for p in base.parameters() if p.requires_grad], "lr": cfg["lr"]},
        {"params": head.parameters(), "lr": cfg["lr"] * 5},
    ], weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)

    history, best = [], {"f1": -1.0}
    for ep in range(cfg["epochs"]):
        base.train(); head.train()
        t0, run_loss = time.time(), 0.0
        for step, (_, x, a, yy) in enumerate(train_loader):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = forward(x.to(dev), a.to(dev))
            loss = loss_fn(logits.float(), yy.to(dev)) / cfg["grad_accum"]
            loss.backward()
            if (step + 1) % cfg["grad_accum"] == 0:
                torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
                opt.step(); opt.zero_grad(set_to_none=True); sched.step()
            run_loss += loss.item()
            if step % 200 == 0:
                print(f"[{cfg['run']} fold {fold}] ep{ep} step {step}/{len(train_loader)} loss {run_loss / (step + 1):.4f}", flush=True)
        vp = predict(va_seqs)
        rec = {"epoch": ep, "loss": run_loss / len(train_loader), "train_s": round(time.time() - t0),
               "val_macro_f1": float(f1_score(y[va_idx], vp.argmax(1), average="macro", labels=[0, 1, 2]))}
        if rec["val_macro_f1"] > best["f1"]:
            best = {"f1": rec["val_macro_f1"], "epoch": ep, "val_probs": vp, "test_probs": predict(te_ids)}
        history.append(rec)
        print(f"[{cfg['run']} fold {fold}] {rec}", flush=True)

    out_dir = f"{VOL}/runs/{cfg['run']}"
    os.makedirs(out_dir, exist_ok=True)
    np.savez(f"{out_dir}/fold{fold}.npz", val_idx=va_idx, val_probs=best["val_probs"], test_probs=best["test_probs"])
    meta = {"fold": fold, "best_epoch": best["epoch"], "best_val_macro_f1": best["f1"], "history": history,
            "gpu": torch.cuda.get_device_name(0), "minutes": round((time.time() - t_start) / 60, 1)}
    Path(f"{out_dir}/fold{fold}.json").write_text(json.dumps(meta, indent=1))
    volume.commit()
    return meta


@app.local_entrypoint()
def main(model: str = "mistral", lr: float = 1e-4, epochs: int = 1, batch_size: int = 16, grad_accum: int = 1,
         max_len: int = 256, lora_r: int = 16, gpu: str = "H100", folds: str = "0,1,2,3,4", seed: int = 42,
         grad_ckpt: bool = False, run: str = ""):
    import sys

    assert model in LLMS
    fold_list = [int(f) for f in folds.split(",")]
    run = run or f"{model}_lora_r{lora_r}_lr{lr:g}_ep{epochs}"
    cfg = dict(model=model, lr=lr, epochs=epochs, batch_size=batch_size, grad_accum=grad_accum, max_len=max_len,
               lora_r=lora_r, n_folds=5, split_seed=42, seed=seed, grad_ckpt=grad_ckpt, run=run, gpu=gpu)
    print("run:", run, json.dumps(cfg))
    print("model dir:", download_llm.remote(model))
    t0 = time.time()
    for res in train_fold.with_options(gpu=gpu).map([cfg] * len(fold_list), fold_list, return_exceptions=True):
        print("fold done:", res if isinstance(res, Exception) else
              {k: res[k] for k in ("fold", "best_epoch", "best_val_macro_f1", "minutes", "gpu")})
    print(f"all folds finished in {(time.time() - t0) / 60:.1f} min")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_modal import aggregate  # local-only import: reuse OOF/test aggregation + validator
    aggregate(run, fold_list, cfg)
