# RESEARCH_V2: tăng điểm từ OOF 0.9758 → ? (deadline 18:00 VN, 2026-10-22)

Ghi chú: **[F]** = fact có nguồn; **[R]** = recommendation của mình; **[U]** = ước lượng chưa kiểm chứng (unverified estimate). Mọi "expected gain" đều là [U], tính trên macro-F1 OOF/LB.
Chi phí Modal giả định [U]: A10G ≈ $1.1/h, L40S ≈ $2/h, H100 ≈ $4/h. Còn ~$15, ~4h.

---

## 0. TL;DR (≤15 dòng)

1. **Pseudo-label test (soft) + retrain 2 model mạnh nhất (mmBERT-base, wolf-defender)**: rẻ, phổ biến nhất trong lời giải Kaggle multilingual (Jigsaw), +0.002–0.005 [U].
2. **mmBERT-base train lâu hơn + LLRD + EMA** (đang still improving ở ep3): +0.001–0.003 single-model [U], rủi ro thấp.
3. **Mistral-7B-Instruct-v0.3 LoRA (`MistralForSequenceClassification`)** làm ensemble member đa dạng nhất, nhắm đúng lỗi BENIGN↔JAILBREAK (cần world-knowledge về "harmful intent"): +0.002–0.006 ensemble [U], ~$5–8, rủi ro trung bình.
4. **Cross-lingual twin check**: data dịch từ cùng nguồn EN → test pt có thể có "anh em" fr/it/en trong train; kNN trên embedding để chuyển nhãn/kiểm tra. Rẻ (vài phút GPU), gain có thể lớn cho pt nếu giả thuyết đúng [U].
5. **Ensemble**: giữ equal-mean (arithmetic) trên probs; thử geometric mean (mean log-prob) và LR stacking chỉ với nested CV; KHÔNG rank-averaging.
6. **FGM** (eps 0.5–1.0) cho 1 run mmBERT-base: +0.000–0.003 [U], 1.5–2× compute; có báo cáo collapse → chỉ dùng nếu OOF tăng.
7. Granite-3.3-2B: **không có `GraniteForSequenceClassification`** trong transformers (docs chỉ có `GraniteModel`, `GraniteForCausalLM`) → cần custom head; ưu tiên Mistral (class có sẵn).
8. Không làm: max_len lớn hơn, per-class/per-language multipliers, rank averaging, fitted weights không nested, TAPT/MLM (thời gian không đủ), LLM reasoning zero-shot, TTA/back-translation, human labelling.

---

## 1. Facts (có link)

### 1.1 Fine-tuning tricks
- **[F] FGM/EMA là trick tiêu chuẩn trong lời giải top Trung Quốc**: lời giải hạng 1 "2023 Global AI Technology Innovation Competition Track 1" liệt kê FGM (noise vào embedding) và EMA (làm mượt weight update) — https://arxiv.org/pdf/2407.01271 . Tổng hợp tricks (FGM, EMA/SWA, R-Drop, TTA, pseudo-label) phổ biến trong cộng đồng CN (Baobao Algorithm Notes, mirror Tencent Cloud): https://cloud.tencent.com/developer/news/620928
- **[F] FGM có thể collapse**: trên Jigsaw, FGM làm BERT/RoBERTa sụp sau 1–3 epoch — https://arxiv.org/pdf/2509.12672 → phải kiểm tra OOF.
- **[F] AWP, distillation, SWA, dropout tuning** là các mục chính trong lời giải hạng 2 (efficiency track) Feedback Prize — https://wandb.ai/darek/fbck/reports/How-To-Build-an-Efficient-NLP-Model--VmlldzoyNTE5MDEx
- **[F] LLRD** (LR lớn ở top layers, nhân hệ số giảm dần mỗi layer) — https://towardsdatascience.com/advanced-techniques-for-fine-tuning-transformers-82e4e61e16e ; xem thêm Zhang et al. "Revisiting Few-sample BERT Fine-tuning" https://arxiv.org/abs/2006.05987
- **[F] R-Drop**: 2 forward pass, KL hai chiều giữa 2 dropout sub-model; hiệu quả trên 18 datasets — https://arxiv.org/pdf/2106.14448v2 (chi phí ~2× forward).
- **[F] Jigsaw Multilingual (Kaggle 2020)**: lời giải top (Lingua Franca notes) dùng XLM-R-large + **pseudo-label** dữ liệu dịch — https://personal-record.onrender.com/post/jigsaw-toxic-comment/ ; top entries ensemble nhiều XLM-R large + monolingual — https://arxiv.org/pdf/2202.11176 ; 3rd place write-up: https://towardsdatascience.com/kaggle-3rd-place-solution-jigsaw-multilingual-toxic-comment-classification-e36d7d194bfb
- **[F] LMSYS Arena (Kaggle 2024)**: hạng 1 dùng **distillation** (teacher lớn 5-fold → student, KL loss); pseudo-labeling và LLM LoRA ensembling là kỹ thuật phổ biến — https://nebius.com/blog/posts/chatbot-arena-competition-review
- **[F] TAPT (continued MLM trên text của task)** cải thiện RoBERTa trên 8 tác vụ classification — https://aclanthology.org/2020.acl-main.740 ; nhưng có nghiên cứu cho thấy TAPT có thể **giảm** điểm với CLS-based FT ở một số task — https://arxiv.org/html/2305.01711v4
- **[F] Granite**: transformers docs (main) chỉ có `GraniteConfig`, `GraniteModel`, `GraniteForCausalLM` — **không có** `GraniteForSequenceClassification` → `AutoModelForSequenceClassification` sẽ không map được GraniteConfig — https://huggingface.co/docs/transformers/main/en/model_doc/granite
- **[F] Mistral seq-cls**: `MistralForSequenceClassification` có sẵn; head `score` khởi tạo mới → LoRA config cần `task_type="SEQ_CLS"`, `modules_to_save=["score"]`, và **phải set `model.config.pad_token_id`** (head pool ở last non-pad token) — https://hf.co/mistralai/Mistral-7B-v0.1/discussions/116 , https://www.digitalocean.com/community/tutorials/mistral-7b-fine-tuning

### 1.2 Ngôn ngữ / dữ liệu dịch
- **[F] mmBERT dùng Gemma 2 tokenizer**; blog ghi nhận yếu hơn ở NER/POS "likely due to tokenizer differences that affect word boundary detection" — https://huggingface.co/blog/mmbert ; paper https://arxiv.org/abs/2509.06888 . XNLI: mmBERT-base 80.5 vs XLM-R-base 78.3 — https://arxiv.org/pdf/2602.21379
- **[F] Translation artifacts**: mismatch giữa text gốc và text dịch (hoặc giữa các kiểu dịch khác nhau) tạo artifacts làm giảm hiệu năng cross-lingual (Artetxe et al., EMNLP 2020) — https://liner.com/review/translation-artifacts-in-crosslingual-transfer-learning ; translate-test/translate-train analysis — https://arxiv.org/pdf/2305.14240
- (Không tìm thấy nguồn riêng về Portuguese.)

### 1.3 BENIGN vs JAILBREAK (adversarial role-play)
- **[F] WildJailbreak** gồm 4 phần: vanilla harmful, adversarial harmful, **adversarial benign** (prompt trông như jailbreak nhưng ý định vô hại, để đo over-refusal), vanilla benign; adversarial-benign và adversarial-harmful dùng **cùng tactic wrapper**, khác ở ý định gốc — https://arxiv.org/pdf/2406.18510
- **[F] WildGuard**: Llama-Guard2 kém hơn nhiều với adversarial prompts so với vanilla; WildGuard huấn luyện trên WildGuardMix (gồm adversarial) mới xử lý được — https://arxiv.org/html/2406.18495v1
- **[F] InjecGuard/PIGuard**: guard models over-defense do bám "trigger words" (NotInject: accuracy tụt ~60%); MOF giảm bias trigger-word bằng ~1000 benign samples chứa trigger words — https://arxiv.org/html/2410.22770v3 , https://huggingface.co/leolee99/PIGuard

### 1.4 Ensemble
- **[F] Forecast combination puzzle**: equal weights thường thắng weights ước lượng do estimation error, nhất là khi sample nhỏ so với số member — https://arxiv.org/pdf/1505.00475 , https://arxiv.org/pdf/2308.05263

---

## 2. Phân tích & khuyến nghị [R]

### 2.1 Vì sao pt khó hơn (giả thuyết, [U])
- pt là 45% data nhưng balance nhãn giống hệt các ngôn ngữ khác → có thể pt được dịch bằng pipeline khác/kém hơn, hoặc nhận phần nguồn khó hơn (WildJailbreak adversarial dài). Kiểm tra nhanh: error rate pt theo độ dài và theo "nguồn" (prompt-injection vs role-play). Nếu lỗi pt tập trung ở role-play dài → vấn đề là nội dung, không phải ngôn ngữ.
- **Language prefix token / language-specific head**: gain kỳ vọng ~0 [U] vì backbone đã nhận diện ngôn ngữ dễ; không ưu tiên.
- **Upweight pt** (loss weight 1.5 cho pt): rẻ, nhưng per-language multipliers đã không giúp → kỳ vọng ≤ +0.001 [U]. Chỉ thử nếu dư GPU.
- **Per-language thresholds**: KHÔNG (cùng lý do per-class multipliers fail ở nested CV).
- **Tokenizer**: Gemma2 (mmBERT) vs SentencePiece-250k (XLM-R) đều phủ pt tốt; khác biệt tokenizer là nguồn **đa dạng** cho ensemble, không phải bug cần sửa.
- **Cross-lingual twin (đáng thử nhất cho pt)**: nếu cùng prompt EN gốc được dịch sang nhiều ngôn ngữ, test pt có thể có bản fr/it/en trong train. Làm: mean-pooled embedding từ mmBERT-base fine-tuned (fold model) cho train+test, cosine kNN cross-language; nếu sim ≥ 0.95 và twin ở train → xem tỷ lệ OOF accuracy khi dùng nhãn twin. Nếu cao hơn model → override/blend. Đồng thời: nếu twins tồn tại, CV hiện tại (không group) bị lạc quan → nhưng LB khớp OOF trong 0.003 nên ảnh hưởng nhỏ.

### 2.2 BENIGN vs JAILBREAK
- Theo WildJailbreak, wrapper (role-play, fictional framing) giống nhau ở cả 2 lớp; khác biệt nằm ở **yêu cầu cốt lõi có hại hay không** → cần world-knowledge về harmfulness. Encoder 300M–560M yếu ở đây; **LLM 7B (Mistral) có prior về harmful content** → member đa dạng nhất cho đúng confusion này [R].
- "More context" (len 768) đã không giúp → không cần thêm length. Có thể thử **head+tail truncation** (giữ 128 đầu + 254 cuối) chỉ khi rảnh — rẻ, gain [U] nhỏ (Sun et al. 2019, https://arxiv.org/abs/1905.05583).
- **LLM reasoning/generative zero-shot**: không — chậm, khó calibrate, khó beat 0.97 fine-tuned trong 2h.

### 2.3 Pseudo-label & distillation: leak notes
- PL **test** rồi thêm vào mỗi fold train: OOF trên train vẫn hợp lệ (test không nằm trong OOF). Rủi ro chính là confirmation bias, không phải leak.
- **Không** dùng OOF probs của train làm soft targets cho student rồi báo OOF của student: teacher của dòng i đã thấy các dòng thuộc val-fold của student → OOF student lạc quan. Nếu distill, dựa vào LB.
- PL được tạo từ ensemble đã thấy toàn bộ train → transductive trên test; thường hợp lệ trên Kaggle (không phải human labelling) — **kiểm tra rules của cuộc thi**.

### 2.4 Ensemble
- Giữ **arithmetic mean of probs, equal weights** làm baseline. Thử **geometric mean** (mean log-probs → softmax) — thường ±0.001 [U].
- **LR stacking trên OOF** (multinomial, C=0.1–1.0, input = log-probs của 6–10 members): chỉ chấp nhận nếu nested CV (outer 5-fold trên OOF) > equal mean ≥ 0.001. Với 14k rows và 30 features thì khả thi nhưng gain thường nhỏ [U].
- **Rank averaging**: không phù hợp multi-class argmax (phá calibration giữa các lớp).
- **Pruning**: thử bỏ mmBERT-small / member yếu nhất; hill-climbing (Caruana) có replacement, nhưng đánh giá nested.

---

## 3. Top-8 action list (xếp theo gain/cost/risk) [R]

| # | Technique | Why (source) | Expected gain [U] | GPU cost Modal [U] | Risk (2h) |
|---|---|---|---|---|---|
| 1 | Pseudo-label test (soft) → retrain mmBERT-base + wolf-defender 5-fold | Jigsaw multilingual top solutions, LMSYS review | +0.002–0.005 (ensemble +0.001–0.003) | 2 × ~40 min L40S ≈ $2.5–3 | Thấp–TB |
| 2 | mmBERT-base 5 ep + LLRD + EMA | still improving ở ep3; LLRD/EMA là tricks chuẩn | +0.001–0.003 single | ~45 min L40S ≈ $1.5 | Thấp |
| 3 | Mistral-7B-Instruct-v0.3 LoRA seq-cls (member mới) | đa dạng nhất; prior harmfulness cho BENIGN↔JAILBREAK (WildJailbreak/WildGuard) | +0.002–0.006 ensemble | 3–5 folds H100 ≈ $5–8 | TB (OOM/pad bugs) |
| 4 | Cross-lingual twin kNN check | dữ liệu dịch từ chung nguồn EN | 0 → +0.01 (nếu twins tồn tại) | ~10 min A10G ≈ $0.2 | Thấp |
| 5 | Ensemble: equal mean vs geometric vs nested LR stacking + prune | forecast combination puzzle | +0.000–0.002 | CPU | Thấp |
| 6 | FGM trên wolf-defender/mmBERT-base | top CN solution (arXiv 2407.01271); có báo cáo collapse | +0.000–0.003 | ~1.7× run ≈ $1.5 | TB |
| 7 | Multi-seed / full-data retrain của best config (thay fold models khi predict test) | Kaggle chuẩn | +0.001–0.002 LB (không đo được OOF) | ~$1–2 | Thấp |
| 8 | XLM-R-large + LLRD 0.95 + EMA (seed mới) | thêm đa dạng tokenizer | +0.000–0.002 ensemble | ~50 min L40S ≈ $1.7 | TB (XLM-R-large hay diverge) |

### Hyperparameters cụ thể
**#1 Pseudo-label**
- Teacher: equal-mean ensemble 6 runs hiện tại trên test.
- Variant A (khuyến nghị): soft targets cho **toàn bộ** test, loss = CE(train hard) + 0.5·KL(student‖teacher soft) trên test rows; hoặc đơn giản: label smoothing kiểu soft-CE với target = teacher probs.
- Variant B (đơn giản hơn): hard PL chỉ những test rows có max prob ≥ 0.95 (thường ~90% test) [U]; giữ tỉ lệ lớp ~62/31/7 (check).
- Mỗi fold: train = 4/5 train + PL test; val = 1/5 train (OOF hợp lệ). Hyperparams như run gốc của model đó.

**#2 mmBERT-base**
- lr 3e-5 (head 1e-4), **LLRD 0.9** (22 layers → bottom ≈ 3e-5·0.9^22 ≈ 3e-6), epochs 5, warmup 10%, cosine/linear decay, wd 0.01, bs 32, max_len 384, bf16.
- **EMA decay 0.999** (step-level), bật sau ~1 epoch, eval & predict bằng EMA weights.
- Tùy chọn: multi-sample dropout (5 dropout heads p=0.1–0.5, mean logits) — rẻ, gain nhỏ [U].

**#3 Mistral-7B LoRA**
- `AutoModelForSequenceClassification.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", num_labels=3, torch_dtype=bfloat16)` (map tới `MistralForSequenceClassification`).
- `tokenizer.pad_token = tokenizer.eos_token` (hoặc `<unk>`), `model.config.pad_token_id = tokenizer.pad_token_id`, padding_side "right" (head lấy last non-pad token; đảm bảo không tự append EOS).
- LoRA: r=16, alpha=32, dropout 0.05, target q,k,v,o,gate,up,down; `task_type="SEQ_CLS"`, `modules_to_save=["score"]`.
- lr 1e-4 (LoRA), score head lr 1e-4–2e-4; 1 epoch (tối đa 2); warmup 5%; bs 16 (8×grad_accum 2); max_len 256–320; gradient checkpointing; H100 bf16 (không cần 4-bit trên H100).
- Nếu ngân sách eo hẹp: **3-fold** (đủ OOF cho ensemble) thay 5-fold, hoặc 1 fold holdout để quyết định weight rồi full-data predict.
- Granite-2B thay thế (rẻ ~3×) nhưng **cần custom head**: `GraniteModel` + pool last non-pad hidden (dim 2048) + `nn.Linear(2048,3)`; hoặc dùng `GraniteForCausalLM` và lấy logits của 3 label tokens (prompt "Label: A/B/C") — rủi ro code cao hơn Mistral.

**#4 Twin kNN**
- Embedding: mean-pool last hidden của 1 fold model mmBERT-base/wolf; L2-normalize; cosine top-1 giữa test và train (khác ngôn ngữ).
- Validate trên train: với mỗi train row, twin trong các fold khác; nếu sim ≥ τ (thử 0.90/0.95/0.98) → accuracy của nhãn twin vs model. Chỉ override khi twin-accuracy > model-accuracy trên subset đó.

**#5 Ensemble**
- Equal arithmetic mean (baseline) | mean log-probs (geometric) | multinomial LR (C ∈ {0.1, 0.3, 1}) trên log-probs, nested 5-fold. Chọn theo nested macro-F1, tie → equal mean.

**#6 FGM**
- Perturb `word_embeddings`, eps 0.5 (thử 1.0), mỗi step: loss + loss_adv; bật từ epoch 2. AWP (adv_lr 1e-4, adv_eps 1e-2, bắt đầu epoch 2) nếu có code sẵn; nếu không → bỏ.

**#7 Seeds/full-data**
- Train best config trên 100% train (+PL), 2–3 seeds, số epoch = epoch tốt nhất ở CV; average với fold models khi predict test.

**#8 XLM-R-large**
- lr 1e-5, LLRD 0.95, 4 ep, warmup 10%, EMA 0.999, bs 16, len 320, seed khác; nếu loss không giảm sau 500 steps → kill (XLM-R-large hay diverge).

### Thứ tự thực thi gợi ý (≈2h GPU, ≈$10–12)
1. (song song, 13:30) #4 twin check (CPU/A10G) + #5 ensemble variants (CPU).
2. (13:30) #2 mmBERT-base LLRD+EMA 5ep **với PL soft (#1)** cùng lúc — gộp để tiết kiệm.
3. (13:30) #3 Mistral 3-fold trên H100 (chạy song song container khác).
4. (15:00) wolf-defender + PL (+FGM nếu #2 cho thấy PL ổn).
5. (16:30) final ensemble nested-CV → submit 2 bản (equal mean all; best nested) trước 17:30.

---

## 4. "Do not do" list [R]
- **Tăng max_len** (768 đã không giúp).
- **Per-class / per-language probability multipliers, per-language thresholds** (đã fail nested CV).
- **Fit ensemble weights trên toàn bộ OOF rồi báo cùng OOF** (overfit; forecast combination puzzle).
- **Rank averaging** cho multi-class.
- **Báo OOF của student distill từ OOF soft labels** như thể không leak.
- **TAPT/MLM continued pretraining** trong 2h: text chỉ ~20k prompts, gain không chắc (arXiv 2305.01711), tốn thời gian setup.
- **Generative LLM zero-shot/CoT reasoning**: chậm, khó calibrate.
- **TTA bằng back-translation/MT**: cần MT model ngoài danh sách backbone → vi phạm rule.
- **RemBERT, distilbert-multilingual**: chậm/yếu, không đáng tiền còn lại.
- **Hard PL với threshold thấp (<0.9)** → confirmation bias trên đúng vùng BENIGN↔JAILBREAK đang sai.
- **Bất kỳ human labelling nào trên test** (cấm).
- **R-Drop** ở giai đoạn này: 2× compute, gain nhỏ so với PL/LLM member.
