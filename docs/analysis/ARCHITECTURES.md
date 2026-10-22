# ARCHITECTURES: vì sao các backbone được phép cho kết quả khác nhau, và tune từng cái thế nào

Ngày 2026-10-22, khoảng 15:50 (giờ VN). Vai trò: architecture & tokenizer analyst. Chỉ dùng CPU, không chạy Modal hay GPU.
Đọc kèm [MODEL_PROFILES.md](MODEL_PROFILES.md), có profile chi tiết theo run, calibration và blend9 + Mistral. File này bổ sung phần **kiến trúc, tokenizer, pretraining**, phần **chẩn đoán Mistral v2**, và **recipe LLM LoRA**.

Nguồn:
- `config.json` và `tokenizer.json` công khai của 8 repo, tải bằng `huggingface_hub` với `token=False`. Không tải weights.
- Tokenize chạy local trên 14,000 text của `train.csv`, đọc trong bộ nhớ từ zip. Không text nào được gửi đi.
- OOF của 14 full-OOF runs, `outputs/mistral_fold0.npz`, và các log `outputs_*.log`.

Scripts nằm trong scratchpad (không commit): `C:\Users\Admin\AppData\Local\Temp\claude\d--A-RMITHACKATHON\1170d8c9-19f8-4773-beaf-7cf8b7cc5e2a\scratchpad\arch_scripts\` (`dl_tok.py`, `fertility.py`, `oof_arch.py`, `extra_arch.py`, `extra2_arch.py`). Bảng thô: `...\scratchpad\arch_out\fertility.csv`, `runs_table.csv`.
Ký hiệu: **[M]** là số đo được bằng các script trên. **[C]** là lấy từ config/tokenizer đã tải. **[L]** là kiến thức từ paper hoặc model card, chưa kiểm lại hôm nay. **[U]** là ước lượng, chưa kiểm chứng.
"Hard slice" = pt, dài hơn 400 chars, nhãn BENIGN hoặc JAILBREAK (n = 3,660). Slice này chứa khoảng 70% lỗi của blend.

---

## TL;DR

1. **Không model đơn nào tốt hơn hẳn.** wolf-defender (mmBERT-base đã fine-tune cho injection) và mmBERT-base hòa nhau:
   - Single run: 0.9671 vs 0.9673. Bootstrap 95% CI của hiệu số là [−0.004, +0.004] [M].
   - Trung bình theo seed: **wolf (3 runs) 0.9713**, mmBERT-base (2 runs) 0.9704, XLM-R-L (2 runs) 0.9681 [M].
   - Nếu buộc phải chọn một backbone encoder thì chọn **wolf**: hard-slice error thấp nhất trong nhóm base-size (6.8% vs 7.4%), và epoch 0 đã đạt 0.948 (mmBERT-base 0.937, XLM-R-L 0.89–0.91).
2. **Truncation không phải nút thắt, đã đo bằng tokenizer thật.**
   - Ở max_len 384, Gemma-2 tokenizer (mmBERT/wolf) chỉ cắt **4.0%** hard slice. Ở 512 chỉ còn 0.9% [M].
   - Các dòng XLM-R cắt ở len320 vẫn khó với cả wolf_len768, dù model này thấy toàn bộ text: 8.3% so với 7.2% trên phần còn lại (+1.1pp). XLM-R ở chính các dòng đó là +2.1–2.6pp. Phần lỗi thật sự do truncation khoảng 1–1.5pp trên 399 dòng, tức **khoảng 4–6 lỗi (≤ +0.0005 F1)** [M].
   - **Kết luận: tăng max_len không phải là đòn bẩy cho encoder.**
3. **Recency của pretraining thấy rõ ở lớp PROMPT_INJECTION.** XLM-R-large (CC-100, 2019) bỏ sót **70–83 PI**, còn mmBERT-base / wolf (corpus 2025) chỉ sót **41–42**. 48/70 PI bị sót có độ dài <200 chars và hầu hết bị đoán thành BENIGN [M].
   - XLM-R lại tốt nhất trên hard pt (family avg 6.4% vs 6.8–7.4%), nhờ có khoảng 3× non-embedding capacity.
   - Hai mặt mạnh/yếu ngược nhau này là lý do XLM-R có ích trong blend.
4. **Diversity theo nguồn gốc** (Jaccard của tập lỗi, cao hơn = giống nhau hơn) [M]:
   - Seed cùng family: 0.41–0.53.
   - Cùng architecture, khác init (mmBERT-base vs wolf): 0.345.
   - Khác family encoder: 0.27–0.30.
   - **Mistral vs encoder (fold 0): 0.21–0.32**, thấp nhất.
   - Thứ tự giá trị cho blend: decoder LLM > encoder khác family > cùng arch khác init > thêm seed.
5. **Mistral v1 (r16, len256, 1 epoch) ổn định trên cả 5 folds**, nhưng folds 1–4 đã bị **dừng tay ở step 600/700** (`outputs_mistral_f1to4.log`). Vì vậy vẫn chưa có full OOF. Mistral có **pt F1 0.9612 trên fold 0, cao nhất trong mọi model**, và chỉ yếu ở PI của fr/it [M].
6. **Mistral v2 (r32, len1024, 2 epochs) đã diverge, không phải do bug pooling hay padding.** Dựng lại loss theo từng đoạn từ log: steps 200–400 có loss **0.97**, xấp xỉ loss 1.01 của một predictor hằng số dưới class weights. Tức model đã collapse ngay sau đỉnh warmup [M].
   - Nguyên nhân chính: bước Adam của head quá lớn. Head LR = 5× = 5e-4 trên hidden state có norm lớn (loss khởi tạo ở folds 1–4 đã là 2.0–2.4 so với ln3 = 1.10).
   - Cộng thêm r32/alpha64 (update LoRA lớn hơn khoảng √2–2×) và chuỗi dài gấp 4.
   - Warmup chỉ 5%. Random shuffle làm padding gấp 3× nên mỗi epoch mất 957 s.
7. **Recipe decoder nên dùng** (chi tiết ở §5):
   - (A) Chạy lại **đúng recipe v1 cho đủ 5 folds**: rẻ (~$2.5) và đã được chứng minh ổn định.
   - (B) Nếu còn H100: **v3 = r16, lr 5e-5, 2 epochs, len512, bs 8×2**, không cần sửa code.
   - Không dùng lại r32 hay len1024. Ngưỡng len512 chọn theo fertility đo được: Mistral cần **1.46× token** so với mmBERT trên pt, và len512 phủ trọn 94% hard slice.
8. **Không nên dùng GPU cho DeBERTa-v3-pi, distilBERT-multilingual, XLM-R-base, Granite.**
   - DeBERTa dùng tokenizer tiếng Anh: 1.98 tok/word trên pt, "instruções" bị cắt thành `in|stru|ções`.
   - distilBERT có [UNK] ở 4.3% text (fr 8.7%).
   - XLM-R-base chỉ đạt 0.88 sau epoch 0.
   - Granite chưa có bằng chứng nào, tokenizer tốn token như Mistral, và nhỏ hơn Mistral 3×.
9. **0.99 không khả thi hôm nay** với approved backbones + train.csv.
   - Blend 14 run hiện có 292 lỗi, trong đó 204 lỗi thuộc hard slice.
   - Mô phỏng: kể cả **sửa hết 100% lỗi hard slice cũng chỉ đạt 0.9881**. Muốn đạt 0.99 thì phải sửa khoảng 75% lỗi hard slice **và** khoảng 50% lỗi còn lại.
   - Mọi recipe đã thử chỉ dịch được hard error trong dải 6.5–8%.
   - Mức thực tế: OOF **0.978–0.981**.

---

## 1. Mỗi backbone khác nhau ở đâu

| | mmBERT-base | **wolf-defender** | mmBERT-small | XLM-R-large | RemBERT | DeBERTa-v3-pi | distilBERT-mult. | Mistral-7B-Inst v0.3 | Granite-3.3-2B-Inst |
|---|---|---|---|---|---|---|---|---|---|
| kiểu | encoder (ModernBERT) | encoder (ModernBERT) | encoder | encoder (RoBERTa) | encoder (BERT) | encoder (DeBERTa-v2) | encoder (6 layers) | decoder causal | decoder causal |
| layers × hidden [C] | 22 × 768, GLU 1152 | 22 × 768 | 22 × 384 [L] | 24 × 1024, FFN 4096 | **32 × 1152**, FFN 4608 | 12 × 768 | 6 × 768 | 32 × 4096, FFN 14336, GQA 8 KV | 40 × 2048, FFN 8192, GQA 8 KV |
| params (tính từ config) | ~307M, trong đó ~110M non-emb | ~307M | ~140M [L] | ~560M, ~303M non-emb | **~575M**, emb chỉ 64M nhờ input emb 256-d | ~184M | ~134M | 7.25B (LoRA r16 ≈ 42M trainable) | ~2.5B |
| position / context [C] | RoPE θ=160k, **8192**; local attn 128 + global mỗi 3 layer | như mmBERT | như mmBERT | **absolute, 512** | absolute, 512 | relative (disentangled, 256 buckets) | absolute, 512 | RoPE θ=1e6, 32k, không sliding window | RoPE θ=1e7, 128k, có muP multipliers |
| head trong `AutoModelForSequenceClassification` [C] | **mean pooling** (`classifier_pooling: mean`) | mean pooling | mean | `<s>` → dense+tanh | [CLS] pooler | [CLS] ContextPooler | [CLS] | custom: hidden của last token (EOS) → Linear(3) | như Mistral |
| tokenizer [C] | Gemma-2 BPE 256k, byte-fallback | **giống hệt mmBERT** (vocab trùng, token ids trùng) | như mmBERT | SPM 250k | SPM-unigram 250k | SPM-unigram 128k (EN), không byte-fallback | WordPiece 119k | BPE 32.8k, byte-fallback | byte-level BPE 49k |
| pretraining [L] | ~3T tokens, 1,800+ ngôn ngữ, annealed language schedule, dữ liệu web 2024–25 (FineWeb2, DCLM…), 2025 | mmBERT + fine-tune binary `benign/injection` (`id2label` [C]) | như mmBERT-base | CC-100 2.5TB, 100 ngôn ngữ, crawl 2018–19 | Wikipedia + Common Crawl, 110 ngôn ngữ, 2020; decoupled emb (input 256 / output 1664 [C]) | English 160GB (DeBERTa-v3 RTD) + fine-tune binary `SAFE/INJECTION` (EN) | distill từ mBERT, Wikipedia 104 ngôn ngữ, 2019 | LLM English-centric + instruct (2024) | 12 ngôn ngữ gồm pt/fr/it (model card), 2025 |
| safety pre-tuning | không | **có (injection)**. Head 2 lớp bị bỏ, khởi tạo lại 3 lớp | không | không | không | có (injection, EN) | không | instruct/safety alignment chung | instruct |

Ghi chú về architecture, có ý nghĩa cho task này:
- **ModernBERT 8k / RoPE không giúp gì ở đây.** Text ngắn: median 67 tokens, p99 669 (mmBERT). Local window 128 đã phủ trọn đa số prompt. Lợi thế thật của mmBERT là **corpus mới hơn và tokenizer hiệu quả** (mục 2–3), không phải context dài.
- **XLM-R và RemBERT bị chặn cứng ở 512.** `train_modal.py` tự cắt về `max_pos − 2`. Như mục 2 cho thấy, giới hạn này không gây thiệt hại đáng kể.
- **RemBERT** tách input embedding (256-d) khỏi output embedding (1664-d, chỉ dùng lúc pretrain). Nhờ vậy embedding nhỏ mà tầng transformer vẫn sâu và rộng nhất (32 × 1152). Đây là encoder lớn nhất trong danh sách.
- **Decoder causal:** chỉ token cuối "thấy" toàn bộ prompt. Vì vậy last-token pooling là lựa chọn đúng. Mean-pooling trên causal states thiên về các token đầu, vốn thấy ít context.
- **wolf vs mmBERT-base:** cùng architecture, cùng tokenizer. Các token đều trùng ids trên câu thử; JSON `pre_tokenizer` chỉ khác cách serialize (`add_prefix_space` so với `prepend_scheme: always`). Mọi khác biệt về kết quả đến từ **init** (đã fine-tune injection) và lr (2e-5 so với 3e-5).

## 2. Tokenizer fertility trên train (đo local)

Tokens/word = tổng token (không tính special) / tổng số từ tách theo khoảng trắng. `trunc@L` = % text có `n_token + 2` lớn hơn L.

**pt (n = 6,300):**

| tokenizer | tok/word | chars/tok | median | p95 | p99 | trunc@256 | @320 | @384 | @512 | @768 | [UNK] / 1k tok |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mmBERT / wolf (Gemma-2) | **1.40** | **4.66** | 136 | 333 | 477 | 12.4 | 5.8 | 2.7 | 0.8 | 0.2 | 0 |
| RemBERT | 1.40 | 4.64 | 136 | 333 | 477 | 12.4 | 6.0 | 2.7 | 0.7 | 0.2 | 0.025 |
| XLM-R | 1.47 | 4.43 | 143 | 346 | 494 | 14.2 | 7.0 | 3.2 | 0.9 | 0.2 | 0.029 |
| distilBERT (WordPiece) | 1.63 | 4.00 | 157 | 386 | 560 | 18.1 | 10.1 | 5.1 | 1.5 | 0.3 | **0.63** |
| DeBERTa-v3 (EN) | 1.98 | 3.29 | 192 | 468 | 679 | 29.8 | 17.0 | 10.2 | 3.3 | 0.6 | 0.042 |
| Mistral v0.3 | **2.04** | 3.19 | 198 | 486 | 700 | 32.2 | 18.4 | 11.2 | 3.8 | 0.7 | 0 |
| Granite-3.3 | 2.03 | 3.20 | 198 | 482 | 694 | 32.5 | 18.5 | 10.9 | 3.7 | 0.6 | 0 |

**Hard slice (pt dài, BEN/JB, n = 3,660):** đây là slice quyết định điểm.

| tokenizer | median | p95 | p99 | trunc@256 | @320 | @384 | @512 | @640 | @768 | @1024 |
|---|---|---|---|---|---|---|---|---|---|---|
| mmBERT / wolf | 180 | 367 | 497 | 19.8 | 9.0 | **4.0** | 0.9 | – | 0.05 | 0 |
| RemBERT | 182 | 360 | 497 | 19.7 | **9.4** | 3.9 | 0.8 | – | 0.05 | 0 |
| XLM-R | 190 | 379 | 516 | 22.9 | **10.9** | 4.7 | 1.1 | – | 0.05 | 0 |
| DeBERTa-v3 | 256 | 512 | 694 | 49.6 | 27.7 | 16.1 | 5.0 | – | 0.6 | 0.1 |
| **Mistral** | 263 | 533 | 721 | **53.9** | 30.1 | 17.8 | **5.8** | 2.2 | **0.8** | 0.1 |
| Granite | 263 | 525 | 714 | 54.2 | 30.2 | 17.4 | 5.7 | – | 0.7 | 0.1 |

Các ngôn ngữ khác [M]:
- tok/word cho fr / it / en: mmBERT 1.48 / 1.55 / 1.28; XLM-R 1.63 / 1.58 / 1.45; Mistral 1.98 / 2.15 / 1.44.
- Trên en, DeBERTa-v3 hiệu quả ngang mmBERT (1.30), nhưng trên pt/fr/it thì đắt gần như Mistral.
- distilBERT có [UNK] ở **4.3%** text: fr 8.7%, en 7.3%. Các ký tự emoji/symbol bị biến thành [UNK], tức mất hẳn thông tin.

Ví dụ tách từ, dùng cụm tự viết (không lấy từ data):
- "instruções anteriores desenvolvedor": mmBERT `▁instruções ▁anteriores ▁desenvol ved or` (5 tokens); Mistral `▁inst ru ções ▁an ter iores ▁desen vol ved or` (10); DeBERTa `▁in stru ções ▁anterior es ▁de sen volve dor` (9).
- distilBERT với "ignore todas as instruções 😀 ✅": `ig ##nore todas as ins ##tru ##ções [UNK] [UNK]`.

Hệ quả:
- mmBERT/wolf và RemBERT có tokenizer tốt nhất cho pt. Mỗi token mang nguyên morpheme, nên embedding đã pretrain có nghĩa ngay.
- Mistral/Granite cần khoảng **1.46× số token** cho cùng một text pt. Max_len của decoder vì vậy phải lớn hơn encoder tương ứng: len256 của Mistral chỉ tương đương khoảng len175 của mmBERT. Đây là lý do probe v1 chỉ thấy trọn 46% hard slice.
- **Chọn max_len cho Mistral theo fertility:** len512 thấy trọn 94.2% hard slice (phần còn lại được head+tail truncation 70/30 xử lý). len640 thấy 97.8%, len768 thấy 99.2%.

## 3. Kết quả đo được theo backbone và giải thích

### 3.1 Bảng chính (full OOF, n = 14,000)

| run | len | OOF | pt | fr | it | en | PI F1 | PI bị sót | hard err % | non-hard err % | epoch-0 val | Δ epoch cuối | loss epoch cuối |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mmbert_small_s1 | 384 | 0.9583 | .9421 | .9757 | .9713 | .9598 | .9610 | 56 | 11.07 | 1.35 | .917 | +.042 | .083 |
| mmbert_small_len768_ep3 | 768 | 0.9604 | .9497 | .9728 | .9705 | .9573 | .9564 | 48 | 9.54 | 1.35 | .913 | +.007 | .017 |
| xlmrl_lr1e-5_ep3 | 320 | 0.9563 | .9483 | .9687 | .9660 | .9402 | .9406 | – | 8.61 | 1.66 | .907 | +.009 | .084 |
| xlmrl_lr1e-5_ep5 | 320 | 0.9629 | .9557 | .9748 | .9731 | .9445 | .9509 | **70** | 7.38 | 1.44 | .907 | +.002 | .013 |
| xlmrl_ep5_s43 | 320 | 0.9626 | .9569 | .9704 | .9687 | .9544 | .9487 | **83** | **6.99** | 1.54 | .888 | +.002 | .018 |
| wolf_lr2e-5_ep3 | 384 | 0.9664 | .9557 | .9794 | .9765 | .9624 | .9599 | 46 | 7.19 | 1.32 | **.948** | +.008 | .008 |
| wolf_lr2e-5_ep4 | 384 | 0.9671 | .9583 | .9796 | .9762 | .9579 | .9621 | 41 | 7.13 | 1.33 | **.947** | +.003 | .001 |
| wolf_len768_s43_ep4 | 768 | 0.9670 | .9590 | .9767 | .9766 | .9594 | .9614 | 46 | 7.30 | 1.27 | .946 | +.001 | .002 |
| wolf_hardw2_ep4 | 512 | 0.9668 | .9607 | .9759 | .9737 | .9563 | – | – | **6.56** | 1.36 | .944 | +.001 | .002 |
| mmbert_base_lr3e-5_ep3 | 384 | 0.9673 | .9556 | .9786 | **.9801** | **.9654** | **.9640** | 42 | 7.87 | **1.12** | .937 | +.010 | .011 |
| mmbert_base_ep5_s43 | 384 | 0.9669 | .9572 | .9789 | .9788 | .9554 | .9656 | 41 | 8.03 | 1.21 | .933 | +.003 | .0001 |
| mmb_pseudo_ep4 (pseudo) | 384 | 0.9690 | .9586 | .9818 | .9827 | .9536 | – | – | 7.32 | 1.11 | .937 | +.001 | .001 |
| wolf_pseudo_ep4 (pseudo) | 384 | 0.9686 | .9605 | .9809 | .9791 | .9532 | – | – | 6.45 | 1.25 | .954 | +.002 | .0005 |

Trung bình theo seed của từng family (mean probs) [M]:

| family | số runs | OOF | pt F1 | hard err % | errors |
|---|---|---|---|---|---|
| wolf | 3 | **0.9713** | 0.9604 | 6.83 | 361 |
| mmBERT-base | 2 | 0.9704 | 0.9595 | 7.38 | 376 |
| XLM-R-L | 2 | 0.9681 | **0.9621** | **6.37** | 365 |
| mmBERT-small | 2 | 0.9641 | 0.9512 | 9.51 | 464 |

Paired bootstrap (1,000 lần) [M]:

| so sánh | Δ | 95% CI | P(Δ > 0) |
|---|---|---|---|
| wolf ep4 − mmBERT-base ep3 | −0.0002 | [−0.0042, +0.0037] | 0.47 |
| wolf avg3 − mmBERT-base avg2 | +0.0009 | – | 0.69 |
| wolf avg3 − XLM-R avg2 | +0.0033 | [−0.0007, +0.0070] | 0.95 |

Runs chưa có OOF:
- RemBERT (đang chạy): val trung bình qua folds là 0.932 sau epoch 0 và 0.954 sau epoch 1 (lịch 3 epochs). Để so sánh, sau epoch 1 XLM-R-L 3ep đạt 0.948 và mmBERT-base 0.957. Ước lượng cuối khoảng 0.961–0.965 [U].
- XLM-R-L + pseudo đã xong: OOF 0.9670 (lạc quan vì dùng pseudo).
- XLM-R-base: 0.882 sau epoch 0 rồi dừng.
- DeBERTa-pi probe: log dừng ngay sau bước load weights, không có số nào.

### 3.2 Vì sao mỗi model ra kết quả như vậy (bằng chứng → giải thích)

**(a) Length và truncation không quyết định.**
- len768 cho Δ ≈ 0: wolf 0.9671 → 0.9670. Kết quả này khớp với số đo: ở len384 chỉ 4% hard slice bị cắt.
- Với 399 dòng hard mà XLM-R cắt ở len320 [M]:

  | model | err% trên các dòng bị XLM-R cắt | err% trên các dòng còn lại |
  |---|---|---|
  | XLM-R-L (hai seed) | 9.27 | 6.7–7.2 |
  | wolf_len768 (thấy trọn text, là đối chứng sạch nhất) | 8.27 | 7.18 |
  | wolf ep4 / ep3 (len384) | 8.77 / 6.52 | 6.93 / 7.27 |
  | mmBERT-base (len384, cắt một phần) | 9.5–10.8 | 7.5–7.9 |

- **Khoảng một nửa độ chênh là do các dòng đó vốn khó (dài), không phải do bị cắt.** Đối chứng wolf_len768 cho +1.1pp, XLM-R cho +2.1–2.6pp. MODEL_PROFILES §2 quy toàn bộ +2.2pp cho truncation. Số liệu ở đây cho thấy phần thật sự do truncation chỉ khoảng 1–1.5pp, tức khoảng 4–6 lỗi.
- Đưa XLM-R lên len384 hay len512 là vô hại và rẻ, nhưng không phải là đòn bẩy.

**(b) Corpus mới hơn giúp mmBERT/wolf hiểu prompt injection.**
- XLM-R-L bỏ sót **70 PI** (seed 43: 83): 48 ca có độ dài <200 chars, 60/70 bị đoán thành BENIGN.
- mmBERT-base sót 42, wolf sót 41.
- PI F1 trên en: XLM-R 0.863, mmBERT 0.916.
- Giải thích [L+U]: CC-100 (crawl 2018–19) có trước thời kỳ LLM, nên gần như không có văn bản kiểu "bỏ qua chỉ dẫn trước…" hay "system prompt". Corpus 2024–25 của mmBERT có loại văn bản này, và wolf còn được fine-tune trực tiếp cho injection.
- Vì macro-F1 cho PI (7% data) trọng số 1/3, lỗi PI tốn khoảng 4× so với lỗi BENIGN.

**(c) Capacity giúp phần ngữ nghĩa "harm" trên hard pt.**
- Theo hard error, family XLM-R-L (khoảng 303M non-embedding) tốt nhất (6.4%), dù tổng điểm thấp hơn.
- Mistral-7B, chỉ 1 epoch và len256, đã có pt F1 tốt nhất trên fold 0 (0.9612). mmBERT-base đạt 0.9520, wolf 0.9584.
- mmBERT-small (khoảng 42M non-embedding) đứng chót trên hard (9.5–11%). Đây là bằng chứng rõ nhất cho việc capacity quan trọng: từ small lên base được +0.009 trên 5/5 folds (MODEL_PROFILES).
- Lỗi BEN↔JB trên pt là lỗi ngữ nghĩa (WildJailbreak: cùng wrapper, khác intent; xem ERRORS.md). Vì vậy model càng biết nhiều về "cái gì là có hại" thì càng tốt.

**(d) Safety pre-tuning giúp wolf khởi đầu nhanh, nhưng mức cuối không cao hơn.**
- Sau epoch 0, wolf đạt 0.945–0.954, mmBERT-base 0.916–0.937.
- Đến epoch 3–4 thì hai bên hội tụ (0.967).
- Head binary bị khởi tạo lại, chỉ backbone giữ lại feature về injection. Phần còn lại của lợi thế là hard error thấp hơn (7.1% vs 7.9%) và diversity với mmBERT (Jaccard 0.345).
- Bù lại, wolf có PI en thấp nhất (0.894). Có thể do khái niệm binary "injection" gộp cả JB lẫn PI [U].

**(e) Lịch học (training curve) khác nhau theo family.**
- mmBERT/wolf bão hòa ở epoch 3–4. Train loss về 0.001–0.0001, tức model đã thuộc lòng train set. Thêm epoch không giúp: mmBERT 5ep 0.9669 so với 3ep 0.9673.
- XLM-R-L học chậm, có thể vì post-LN, lr 1e-5 bảo thủ và kích thước lớn. Model này vẫn còn tăng ở epoch 5 (+0.002, loss 0.013), nên là family **duy nhất còn under-trained**. Ở seed averaging, XLM-R cũng hưởng lợi nhiều nhất: +0.0052 khi gộp 2 seed, so với +0.003 ở các family khác.

**(f) Diversity theo nguồn gốc** [M]:

| cặp | Jaccard lỗi | Yule Q | F1 khi gộp cặp | gain so với model tốt hơn |
|---|---|---|---|---|
| wolf ep3 vs ep4 (cùng seed) | 0.531 | 0.992 | 0.9698 | +0.0027 |
| wolf s42 vs s43 / len768 | 0.468 | 0.988 | 0.9704 | +0.0033 |
| XLM-R s42 vs s43 | 0.471 | 0.988 | 0.9681 | +0.0052 |
| mmBERT-base s42 vs s43 | 0.409 | 0.982 | 0.9704 | +0.0032 |
| mmBERT-base vs wolf (cùng arch, khác init) | 0.345 | 0.973 | 0.9713 | +0.0041 |
| mmBERT-base vs XLM-R | 0.298 | 0.962 | 0.9709 | +0.0036 |
| wolf vs XLM-R | 0.290 | 0.960 | **0.9724** | +0.0053 |
| **Mistral vs mmBERT-base (fold 0)** | **0.212** | 0.923 | – | – |
| Mistral vs wolf / XLM-R (fold 0) | 0.27 / 0.32 | 0.956 / 0.965 | – | – |
| tham chiếu fold 0: mmBERT seed vs seed | 0.457 | 0.985 | – | – |

Trên hard slice ở fold 0:

| cặp | Yule Q | Jaccard |
|---|---|---|
| Mistral vs encoder | **0.77–0.89** | 0.19–0.28 |
| encoder khác family | 0.90–0.95 | 0.30–0.40 |
| encoder cùng family, khác seed | 0.97–0.985 | 0.51–0.58 |

Mistral là member mắc lỗi ở *chỗ khác* nhiều nhất trên đúng slice khó nhất.

## 4. Mistral-7B LoRA: chuyện gì đã xảy ra

### 4.1 v1 (r16, len256, 1 epoch, lr 1e-4, bs16, không dùng grad-ckpt)

- **Fold 0 (n = 2,800):**

  | model | macro-F1 | pt | fr | it | en | PI F1 | hard err |
  |---|---|---|---|---|---|---|---|
  | Mistral v1 | 0.9638 | **0.9612** | 0.9673 | 0.9661 | 0.9570 | 0.9547 | 7.58% |
  | wolf ep4 | 0.9707 | 0.9584 | 0.9870 | 0.9831 | – | 0.9714 | 6.89% |

  NLL 0.105, so với 0.214 của mmBERT. Mistral ít over-confident hơn hẳn.
- **Điểm yếu:** PI của fr. PI F1 fr 0.925 so với 0.970 của mmBERT; 14 PI bị đoán thành BEN, mmBERT chỉ 10. Train loss ở 100 steps cuối khoảng 0.11, so với 0.001 của encoder. Model rõ ràng **under-trained**.
- **Truncation ở len256 không làm hỏng Mistral:**
  - Hard rows bị Mistral cắt: Mistral sai 7.71%. Cũng trên các dòng đó, wolf (thấy trọn) sai 8.74%, mmBERT 9.00%, XLM-R 6.68%.
  - Hard rows không bị cắt: Mistral 7.42%, wolf 4.75%.
- **Blend trên fold 0:**
  - blend 12 run non-pseudo + 0.2 × Mistral: lỗi 65 → 58, pt F1 0.9661 → 0.9713. Macro-F1 gần như không đổi (0.9751 → 0.9750) vì thêm 1–2 lỗi PI.
  - Với blend9, MODEL_PROFILES đo được +0.0018 khi thêm Mistral làm member thứ 10.
- **Folds 1–4 của đúng recipe này đã chạy và ổn định.** Running loss ở step 600/700 là 0.336–0.384, rất giống fold 0. Tuy vậy chúng **bị dừng bằng tay trước khi xong** ("Stopping app - user stopped from CLI"), nên không có file npz nào được ghi.

### 4.2 v2 (r32, alpha64, len1024, 2 epochs, bs 8×2, grad-ckpt) đã diverge

Dựng lại loss trung bình theo từng đoạn từ running mean trong log (`outputs_mistral_v2.log`):

| micro-steps | v2 loss từng đoạn | v1 loss từng đoạn (cùng fold 0) |
|---|---|---|
| 0–200 | 0.44 | 0.49 |
| **200–400** | **0.97** | 0.30 |
| 400–600 | 0.62 | 0.20 |
| 600–800 | 0.46 | 0.11 (600–700, hết epoch) |
| 1000–1400 | 0.39–0.40 | – |
| epoch 1: 200–600 | 0.33 | – |

- **0.97 gần bằng 1.01**, là weighted-CE của predictor hằng số tốt nhất dưới class weights `freq^-0.5`. Như vậy ở optimizer steps khoảng 100–200, ngay sau đỉnh warmup (70 steps = 5%), model đã **collapse về output hằng số**. Sau đó model chỉ phục hồi một phần.
- Val 0.47 sau epoch 0 khớp với một model phân biệt BEN/JB được phần nào nhưng gần như không bao giờ đoán PI.
- Dừng run ở epoch 1 là đúng: loss vẫn ở 0.33, gấp 3 lần v1.

**Nguyên nhân, xếp theo mức độ chắc chắn:**
1. **Bước Adam của head quá lớn.**
   - Head `Linear(4096, 3)` khởi tạo ngẫu nhiên, LR = 5 × lr = **5e-4**.
   - Loss tại step 0 ở folds 1–4 của v1 là **1.98 / 2.43 / 2.03 / 2.11**, cao hơn hẳn ln3 = 1.10. Logit lúc khởi tạo đã có độ lớn khoảng 2–3, tức hidden của token cuối có norm lớn (ước khoảng ‖h‖₂ ~ 10², [U]).
   - Adam cập nhật gần như theo dấu gradient, mà gradient của head = δ·h. Vì vậy mỗi bước đẩy logit một lượng khoảng lr·‖h‖₁ theo cùng một hướng: **cỡ O(1) mỗi step ở 5e-4** [U, ước lượng bậc độ lớn]. v1 sống sót được, nhưng không có biên an toàn.
2. **r32 / alpha64.** Tỉ lệ alpha/r vẫn là 2, nhưng rank gấp đôi, nên ΔW = s·B·A lớn hơn khoảng √2–2× với cùng lr (đây là lập luận của rsLoRA [L]).
3. **Chuỗi dài gấp 4 (1024).** Gradient attention qua context dài có variance lớn hơn. Clip 1.0 là global trên toàn bộ LoRA + head nên không chặn riêng được head.
4. **Warmup 5%** quá ngắn cho 7B. Không có eval giữa epoch hay cơ chế abort, nên mất 37 phút H100 mới biết run hỏng.
5. **Hiệu năng, không phải nguyên nhân diverge.** Random shuffle với bs8 ở len1024 tạo ra **6.41M token đã pad, so với 2.16M token thật (3×)**. Cộng thêm grad-ckpt (+~30%), mỗi epoch mất 957 s, so với 342 s của v1.

**Không phải nguyên nhân** (đã kiểm code `modal_jobs/train_lora.py`):
- **Padding side:** code pad bên phải và lấy `last = a.sum(1) - 1`, đúng với right padding. Mask dựng từ độ dài, không từ pad id, nên pad = eos (Mistral không có pad token) vô hại.
- **EOS:** append `</s>` (id 2) cho cả train và test. Token này là vị trí pooled nhất quán.
- **bf16:** base bf16, head fp32, logits ép `.float()` trước CE. PEFT mặc định autocast adapter sang fp32 [L].
- **Lưu ý nếu đổi sang `MistralForSequenceClassification` của HF:** model này dò pad theo `pad_token_id`. Khi pad = eos và có EOS ở cuối, nó sẽ pool token *trước* EOS. Head custom hiện tại tránh được chuyện này.

## 5. Cấu hình decoder LoRA nên dùng để vượt encoder trên pt role-play dài

| knob | khuyến nghị | lý do |
|---|---|---|
| pooling | **last token = EOS, right padding, gather `a.sum-1`** (giữ như hiện tại) | causal: chỉ token cuối thấy toàn bộ prompt. Mean-pool kém hơn |
| r / alpha | **r16, alpha 32** | ổn định trên 5/5 folds (v1). Nếu tăng r thì giữ alpha = r (scale 1) hoặc dùng rsLoRA, và hạ lr |
| lr | **1e-4 cho 1 epoch**; **5e-5 nếu 2 epochs** | v1 ổn định. Với 2 epochs, lr thấp hơn giữ tích phân LR tương đương mà đỉnh an toàn hơn. Head LR hiện bị hard-code = 5×lr; muốn hạ nó chỉ có cách hạ lr |
| epochs | 1 (an toàn) hoặc 2 (đáng thử nếu còn H100) | v1 kết thúc ở loss 0.11, còn dư địa. LLM thường đạt tốt nhất ở 1–2 epochs; code tự chọn epoch tốt nhất theo val |
| max_len | **512** | theo fertility đo được: phủ trọn 94% hard slice, phần còn lại head+tail 70/30. len768 thêm 5% coverage, nhưng bs8 × 768 không có ckpt sát giới hạn 80GB. len1024 lãng phí |
| batch | **bs 8 × grad-accum 2** (effective 16, như v1) | micro-batch 8 × 512 = 4,096 tokens, đúng peak của v1 (16 × 256) đã chạy được không cần ckpt |
| grad-ckpt | **tắt** | không cần ở 4,096 tokens/micro-batch; ckpt làm chậm khoảng 30% |
| class weights | giữ `freq^-0.5` | giống encoders, để blend calibration nhất quán; không có bằng chứng nên đổi |
| warmup | 5% (hard-code). Nên là 10% nếu sửa code | xem §4.2 |
| precision | bf16 base, LoRA và head fp32 (giữ). **Không dùng fp16** | Mistral có activation lớn, fp16 dễ overflow |
| eos / bos | giữ `<s> … </s>` | nhất quán giữa train và test |
| batching | nếu được sửa code: **length-bucketed shuffle** | ở len512 bs8: 5.41M token đã pad so với **2.03M** token thật. Nhanh hơn khoảng 2.5× |
| prompt template | tùy chọn, [U]: bọc text trong `[INST] Classify … [/INST]` | đưa vị trí pooled về một "decision token" quen thuộc với model instruct. Chưa kiểm chứng |

**Lệnh, xếp theo thứ tự ưu tiên** (Windows PowerShell: chạy `$env:PYTHONUTF8=1` trước):

```bash
# A. Recipe v1 đủ 5 folds (đã chứng minh ổn định; ~$2.5, ~12–15 min wall với 5×H100).
#    Tên run mới để không ghi đè summary của probe.
python -m modal run modal_jobs/train_lora.py --model mistral --gpu H100 --lr 1e-4 --epochs 1 --batch-size 16 --max-len 256 --lora-r 16 --seed 42 --folds 0,1,2,3,4 --run mistral_r16_len256_ep1_5f

# B. v3, không cần sửa code: len512 theo fertility, 2 epochs với lr thấp (~$7–9, ~25–30 min wall).
#    Chỉ chạy nếu còn 5 H100 rảnh và launch được trước ~16:20.
python -m modal run modal_jobs/train_lora.py --model mistral --gpu H100 --lr 5e-5 --epochs 2 --batch-size 8 --grad-accum 2 --max-len 512 --lora-r 16 --seed 42 --folds 0,1,2,3,4 --run mistral_r16_len512_ep2_lr5e-5
```

Cách theo dõi B:
- Kỳ vọng mỗi 200 micro-steps loss giảm đều, và trong epoch 0 không có đoạn nào > 0.6 sau step 400. Nếu running loss *tăng* giữa hai lần in thì abort.
- Cuối epoch 0, val fold phải ≥ 0.95. Nếu không đạt thì dừng ngay.

Bản rẻ hơn cho A: chỉ chạy `--folds 1,2,3,4 --run mistral_lora_f0_probe` (fold 0 đã có trên Volume). Sau đó gộp bằng `python -m modal run modal_jobs/train_modal.py --run mistral_lora_f0_probe --aggregate-only --folds 0,1,2,3,4`. Lưu ý: `summary.json` khi đó sẽ ghi cfg mặc định của train_modal (model=wolf). Đây chỉ là lỗi hiển thị, nhưng dễ đọc nhầm.

Patch tùy chọn, **CHƯA test**, chỉ áp dụng sau khi có A và B: nhỏ, tất cả trong `train_lora.py`.
1. `head` zero-init (`torch.nn.init.zeros_(head.weight); zeros_(head.bias)`) để bắt đầu đúng tại uniform.
2. Chuẩn hóa pooled vector trước head (`pooled = pooled / pooled.norm(dim=-1, keepdim=True) * hidden**0.5`, hoặc một `LayerNorm(hidden)` fp32).
3. Head LR 1× thay vì 5×.
4. Warmup `0.1 * steps`.
5. Length-bucketed sampler: permute, chia mega-batch `bs*64`, sort theo độ dài trong mỗi mega-batch, cắt thành batch, rồi shuffle thứ tự batch.
6. Eval mỗi nửa epoch.

Với 1–4, r32 và lr 1e-4 lẽ ra sẽ ổn định [U]. Với 5, thời gian mỗi epoch ở len512 giảm khoảng 2.5×.

**Granite-3.3-2B:**
- Tokenizer tốn token như Mistral (2.03 tok/word trên pt).
- Nhỏ hơn Mistral 3×, kiến trúc 40 × 2048 với muP multipliers (embedding ×12, residual ×0.22).
- Code dùng `bos = eos = pad = <|end_of_text|>` (id 0). Với cách pool bằng mask, điều này vẫn đúng.
- Chưa có bằng chứng nào cho thấy nó hơn Mistral. Chỉ chạy nếu còn L40S rảnh **sau khi** A đã cho full OOF:

```bash
python -m modal run modal_jobs/train_lora.py --model granite --gpu L40S --lr 1e-4 --epochs 2 --batch-size 8 --grad-accum 2 --max-len 512 --lora-r 16 --run granite_r16_len512_ep2
```

## 6. Khuyến nghị theo từng model (encoders)

| model | trạng thái | khuyến nghị | lệnh (nếu có) |
|---|---|---|---|
| **wolf-defender** | đã bão hòa. 3 runs ≈ 0.967 mỗi run, avg 0.9713 | **Dừng tune.** EMA 0.998 làm hỏng epoch 0 (cửa sổ khoảng 500 steps, dài hơn 1 epoch 350 steps). FGM tốn 2× thời gian. len768 và hard-weight đều là noise ở tổng điểm (hardw2 có hard err thấp nhất 6.56% nhưng non-hard tăng). Nếu chỉ còn A10G rảnh, 1 seed mới là rẻ nhất, nhưng gain cho blend ≈ 0 | (tùy chọn) `python -m modal run modal_jobs/train_modal.py --model wolf --lr 2e-5 --epochs 4 --batch-size 32 --max-len 384 --seed 71 --run wolf_ep4_s71` |
| **mmBERT-base** | đã bão hòa từ epoch 3 (loss 1e-4 ở epoch 5) | chờ `mmb_ls01_lr5e-5_ep4` đang chạy. Không chạy thêm | – |
| **XLM-R-large** | **under-trained** (epoch 5 vẫn +0.002, loss 0.013); seed-variance cao nhất; tốt nhất trên hard pt; yếu ở short PI | thêm epochs và 1 seed mới. Dùng len384: truncation chỉ đáng khoảng 4–6 lỗi, len384 đã giảm hard trunc từ 10.9% xuống 4.7% và rẻ hơn 512 | `python -m modal run modal_jobs/train_modal.py --model xlmr-large --gpu L40S --lr 1e-5 --epochs 7 --batch-size 16 --max-len 384 --seed 81 --run xlmrl_ep7_len384_s81` |
| **RemBERT** | đang chạy (lr 1.5e-5, 3 epochs, bs 8×2, len320). Epoch 1 đạt 0.954 | chờ kết quả, rồi làm nested addition test. Nếu OOF ≥ 0.962 và có ích trong blend, run sau: 4 epochs, len384 (hard trunc 9.4% → 3.9%) | sau khi xong: `python -m modal run modal_jobs/train_modal.py --model rembert --gpu L40S --lr 1.5e-5 --epochs 4 --batch-size 8 --grad-accum 2 --max-len 384 --seed 44 --run rembert_ep4_len384_s44` |
| mmBERT-small | trần capacity (hard err 9.5–11%) | không chạy thêm. Giữ trong blend chỉ khi nested CV ủng hộ | – |
| DeBERTa-v3-pi | chưa có số liệu | **không chạy.** Tokenizer EN (pt 1.98 tok/word, cắt thành sub-word vô nghĩa), pretraining English-only, chỉ 12 layers | – |
| distilBERT-mult. | – | **không chạy.** 6 layers, [UNK] ở 4.3% text | – |
| XLM-R-base | epoch 0: 0.882 | **không chạy.** Bị XLM-R-large lấn át hoàn toàn | – |

Ngân sách: A (~$2.5) + XLM-R 7ep (~$3.5) + B (~$8) ≈ $14, nằm trong $21 của account 2. Nếu H100 bị giới hạn, ưu tiên A > XLM-R > B.

## 7. 0.99 có khả thi không?

Blend 14 runs equal mean (OOF): **0.9764, 292 lỗi, trong đó 204 lỗi thuộc hard slice**. Mô phỏng sửa ngẫu nhiên một phần lỗi (trung bình 20 lần) [M]:

| sửa bao nhiêu lỗi | OOF macro-F1 |
|---|---|
| 25% hard | 0.9793 |
| 50% hard | 0.9823 |
| 100% hard, 0% còn lại | **0.9881** |
| 75% hard + 50% còn lại | 0.9911 |
| 80% hard + 60% còn lại | 0.9928 |

- Để đạt 0.99 phải sửa khoảng **75% lỗi hard và khoảng 50% lỗi PI / non-pt cùng lúc**. Nhóm lỗi thứ hai khó vì macro-F1 cho lớp PI (7%) trọng số 1/3.
- Hard error của mọi recipe đã thử, kể cả Mistral fold 0 (7.6%), đều nằm trong dải 6.5–8%. Khoảng 50 lỗi bị *mọi* model đoán sai, nhiều khả năng là label noise (ERRORS.md).
- Mức thực tế trước 18:00: **OOF khoảng 0.978–0.981**. Điều kiện là Mistral 5-fold xác nhận được tín hiệu của fold 0, và XLM-R 7ep cùng RemBERT có ích trong nested test. Dự báo public khoảng 0.975–0.978 (gap OOF→public trước đây khoảng −0.003).
- Ai trên LB đạt ≥ 0.99 thì nhiều khả năng đang dùng thứ gì đó ngoài approved + train.csv. Điều này không kiểm chứng được và không phải hướng nên theo.

## 8. Files

- `docs/analysis/ARCHITECTURES.md` (file này).
- Scripts: `C:\Users\Admin\AppData\Local\Temp\claude\d--A-RMITHACKATHON\1170d8c9-19f8-4773-beaf-7cf8b7cc5e2a\scratchpad\arch_scripts\`
- Kết quả thô: `C:\Users\Admin\AppData\Local\Temp\claude\d--A-RMITHACKATHON\1170d8c9-19f8-4773-beaf-7cf8b7cc5e2a\scratchpad\arch_out\` (`fertility.csv`, `runs_table.csv`, `ntok_*.npy`)
- Tokenizer và config tải về (công khai, không có weights): `...\scratchpad\arch_tok\<model>\`
