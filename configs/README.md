# Run configs

Each run = one copy of the `CFG` dict at the top of the Kaggle notebook. Record the exact dict in
`experiments/log.csv` (notes column) and keep `val_size=0.15, seed=42` identical for runs you plan to blend.

| run_name | model_slug | lr | max_len | batch | epochs | notes |
|---|---|---|---|---|---|---|
| s1_mmbert_small | mmbert-small | 5e-5 | 256 | 32 | 2 | first valid submission |
| xlmr_large | xlm-roberta-large | 1e-5 | 256 | 16 | 3 | watch for all-BENIGN collapse |
| wolf_defender | wolf-defender-prompt-injection | 2e-5 | 256 | 32 | 3 | mmBERT-base already tuned for injection |
| mmbert_base | mmbert-base | 3e-5 | 512 | 16 | 3 | long context |
