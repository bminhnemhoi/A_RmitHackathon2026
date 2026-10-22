# RESEARCH — Challenge 1: BENIGN / JAILBREAK / PROMPT_INJECTION (macro-F1)

> Ngày: 2026-10-22, deadline 18:00 ICT. Research time-box ~25 phút.
> Quy ước: **[F]** = fact có nguồn (link ở mục Sources). **[R]** = khuyến nghị của tôi. **[UL]** = ước lượng chưa kiểm chứng (back-of-envelope), phải đo lại trên Kaggle.

---

## 0. Phát hiện quan trọng nhất (đọc trước)

1. **[F] `patronus-studio/wolf-defender-prompt-injection` được fine-tune từ `jhu-clsp/mmBERT-base`** (ModernBERT, tokenizer Gemma 2, vocab 256k, 22 layers, hidden 768). Đây là binary classifier `0=benign, 1=injection` ("prompt injection or jailbreak-like instruction"), train trên NotInject, deepset, jayavibhav, WildJailbreak, UltraChat... (nguồn S3, S4).
   → **[R]** Đây là init tốt nhất trong list: cùng backbone multilingual với mmBERT-base nhưng đã "biết" domain injection/jailbreak. Load bằng `AutoModelForSequenceClassification.from_pretrained(..., num_labels=3, ignore_mismatched_sizes=True)`: chỉ `classifier` (Linear 768→2, không bias) bị thay; `head` (dense+act+norm) vẫn giữ (theo cấu trúc `ModernBertForSequenceClassification`, S7).
2. **[F] GOTCHA nghiêm trọng với wolf-defender trên transformers 4.x:** config của wolf-defender được lưu bằng `transformers_version: 5.10.2`, dùng format mới `rope_parameters: {full_attention: {rope_theta:160000}, sliding_attention: {rope_theta:160000}}` và **không có** key `global_rope_theta` / `local_rope_theta` (S4-config). Trong transformers 4.48, `ModernBertConfig` mặc định `global_rope_theta=160000`, **`local_rope_theta=10000`** (S7-config). mmBERT gốc dùng `local_rope_theta=160000` (S2-config).
   → **[R]** Nếu Kaggle chạy transformers 4.x: phải set tay `config.local_rope_theta = 160000.0; config.global_rope_theta = 160000.0` trước khi load, nếu không các sliding-attention layer sẽ dùng RoPE sai → **suy giảm âm thầm, không báo lỗi**. Cách an toàn: `cfg = AutoConfig.from_pretrained(path); cfg.local_rope_theta=160000.0; cfg.global_rope_theta=160000.0; cfg.num_labels=3; model = AutoModelForSequenceClassification.from_pretrained(path, config=cfg, ignore_mismatched_sizes=True)`. Luôn `print(transformers.__version__)` và `print(model.config)`. (Tôi chưa test trên transformers 5.x xem key cũ của mmBERT có được convert đúng không — cũng nên in config kiểm tra.)
3. **[F] `protectai/deberta-v3-base-prompt-injection-v2`: English-only, "does not detect jailbreak attacks or handle non-English prompts", project archived** (S5). Tokenizer DeBERTa-v3 (English). → **[R]** ưu tiên thấp nhất cho data PT/FR/IT.
4. **[F] transformers (kể cả main) KHÔNG có `GraniteForSequenceClassification`** — `MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES` không chứa `granite`; Mistral thì có (`MistralForSequenceClassification`) (S8, tôi đã grep file `modeling_auto.py` trên main). → Granite cần custom head (last-token pooling + Linear) — tốn thời gian code, rủi ro cao với deadline hôm nay.
5. **[F] Định nghĩa nhãn chuẩn (Meta Prompt-Guard, cùng 3 nhãn BENIGN/INJECTION/JAILBREAK, eval trên EN/FR/DE/HI/IT/PT/ES/TH):** INJECTION = lệnh/instruction "out of place" nhúng trong dữ liệu (thường third-party); JAILBREAK = cố ý override system prompt / safety conditioning (thường ở user input); **nếu injection dùng kỹ thuật jailbreak trực tiếp → gán JAILBREAK** (S6). Rất có thể dataset cuộc thi theo convention tương tự (chưa kiểm chứng — hãy đọc sample PROMPT_INJECTION trong train).

---

## 1. Tóm tắt khuyến nghị (top 5, xếp theo gain/cost)

1. **[R] Train wolf-defender → 3 class (fp16, sdpa, max_len 256–384, lr 2e-5, 3 epoch, stratified 5-fold hoặc 1 holdout 15%)** — init domain-specific + multilingual; rẻ (~ cỡ base model). Nhớ fix RoPE (mục 0.2).
2. **[R] Hậu xử lý tối ưu macro-F1: nhân xác suất theo class `p_c * w_c` (grid search w trên OOF/validation), đặc biệt tăng w cho PROMPT_INJECTION (7%)** — gần như miễn phí, thường là gain lớn nhất cho macro-F1 khi có class hiếm (S12, S13, S14). Tránh overfit: tối ưu trên OOF đủ lớn, grid thô.
3. **[R] Ensemble softmax (trung bình prob) của 2–3 model khác tokenizer: wolf-defender/mmBERT-base + XLM-R-large (+ mmBERT-small hoặc XLM-R-base nếu dư thời gian)** — đa dạng tokenizer (Gemma2 vs SentencePiece XLM-R) thường cho ensemble gain. Áp dụng class-weight tuning SAU khi ensemble.
4. **[R] XLM-R-large: lr 1e-5, warmup 10%, fp16, bs 16 (hoặc 8×2 accum), 3 epoch, grad clip 1.0, chạy ≥2 seed vì dễ collapse** (S9, S10). Kiểm tra sau epoch 1: nếu macro-F1 ≈ chỉ predict BENIGN → collapse, restart seed khác.
5. **[R] Loss: CE + class weights nhẹ (vd. sqrt inverse freq) hoặc CE thường + threshold tuning; label smoothing 0.05 tùy chọn.** Focal/R-Drop/LLRD là "nice-to-have", chỉ thêm khi baseline đã chạy ổn — với deadline hôm nay, ưu tiên ensemble + threshold hơn là tune loss.

**Không khuyến nghị hôm nay:** Mistral-7B LoRA (quá chậm trên T4, [UL] nhiều giờ/epoch), Granite-2B (không có seq-cls class sẵn, cần code head riêng), protectai DeBERTa (English-only), RemBERT (bs 8, chậm, lợi ích chưa rõ).

---

## 2. Bảng đề xuất

| Đề xuất | Lý do / nguồn | Chi phí GPU/thời gian | Trong list duyệt? | Rủi ro |
|---|---|---|---|---|
| wolf-defender → 3-class fine-tune | [F] base = mmBERT-base, đã train injection+jailbreak binary, multilingual tokenizer (S3, S4) | [UL] tương đương mmBERT-base: ~10–20 phút/epoch trên 1×T4 ở len 256 với dynamic padding | Có | RoPE config v5 vs v4 (mục 0.2); card chỉ validate EN/DE, PT/FR/IT ít data (S3) |
| mmBERT-base từ scratch (không wolf) | [F] XNLI: mmBERT-base 77.1 vs XLM-R-base 74.6 (S1 Table 3) | [UL] như trên | Có | Giống wolf về gotcha FA2/compile; có thể dư thừa nếu đã có wolf |
| XLM-R-large | Encoder multilingual mạnh, tokenizer khác → đa dạng ensemble. [F] paper mmBERT không so với XLM-R-large (S1) | [UL] ~3× base; ước tính 15–30 phút/epoch ở len 256 trên 1×T4 fp16 — **đo 50 step đầu rồi nhân** | Có | Collapse/instability (S9, S10); OOM nếu bs lớn → bs 8 + accum 2 hoặc grad checkpointing |
| Class-prob scaling / threshold trên OOF | [F] per-class threshold là cách hiệu quả nhất cho macro-F nhưng dễ overfit class hiếm (S12); class weights như threshold đa lớp (S13); logit adjustment theo prior (S14) | CPU, vài giây | N/A | Overfit validation → dùng OOF 5-fold hoặc grid thô (bước 0.05–0.1) |
| Ensemble mean softmax + multi-seed | Practice chuẩn; giảm variance (S9 khuyên chạy nhiều seed) | Tuyến tính theo số model | Có | Hết giờ; cần lưu OOF + test probs mỗi model |
| Head+tail truncation | [F] Sun et al.: head 128 + tail 382 tốt nhất trên IMDb/Sogou (S11) | Không tốn thêm | N/A | Text mean 294 chars → phần lớn ngắn; chỉ ảnh hưởng đuôi dài (max 18,550 chars). Payload injection hay nằm ở cuối text dài → tail quan trọng |
| LLRD (decay 0.9–0.95) | [F] ξ=0.95, lr 2e-5 tốt nhất trong Sun et al. (S11) | Không tốn thêm | N/A | Gain nhỏ, cần code param groups |
| R-Drop | [F] KL giữa 2 forward pass có dropout (S15) | ~2× compute/step | N/A | ModernBERT/mmBERT config có dropout = 0.0 hết (S2-config) → R-Drop vô nghĩa trừ khi bật dropout; chỉ hợp với XLM-R |
| Focal loss | [F] Lin et al. 2017 (S16) | Không tốn thêm | N/A | Thêm 1 hyperparam γ; tương tác với threshold tuning — không ưu tiên |
| Granite-3.3-2B LoRA + custom head | [F] hỗ trợ EN/DE/ES/FR/JA/PT/AR/CS/IT/KO/NL/ZH, 128K ctx (S17); không có seq-cls class (S8) | [UL] 4-bit QLoRA ~1h+/epoch trên T4 | Có (LoRA) | Code head riêng, fp16 vs bf16 weight issues trên T4 (S18), chậm |
| Mistral-7B LoRA | Có `MistralForSequenceClassification` (S8) | [UL] vài giờ/epoch trên T4 4-bit | Có (LoRA) | Không kịp deadline |
| protectai DeBERTa-v3 PI v2 | [F] English-only, không detect jailbreak, archived (S5) | Rẻ | Có | Tokenizer English → kém cho PT/FR/IT |

---

## 3. Hyperparameters đề xuất theo model

Fact tham chiếu:
- [F] mmBERT paper (fine-tune GLUE/XTREME): lr grid {2e-5…8e-5}, đa số best ở **2e-5 hoặc 3e-5**; epochs {1,2,3,5,10}; **bs 32; warmup ratio 0.06** (S1).
- [F] mmBERT model card ví dụ XNLI: lr 3e-5, 3 epoch, bs 32, wd 0.01, max_length 512 (S2).
- [F] wolf-defender training: lr **1e-5**, 3 epoch, effective bs 32 (8×4), wd 0.01, bf16, + SupCon loss 0.05 + FreeLB (S3).
- [F] RoBERTa/XLM-R: lr 1e-5 ổn định hơn 3e-5; warmup 6% (RoBERTa paper) / 10% (S9, S10).

Khuyến nghị [R] (chưa tune trên data này — điểm xuất phát):

| Model | lr | Epochs | Batch (T4 16GB, fp16) | max_len | Warmup | WD | Ghi chú |
|---|---|---|---|---|---|---|---|
| mmBERT-small (140M) | 5e-5 (thử 3e-5) | 3–4 | 32 | 256–512 | 0.06 | 0.01 | Rẻ, dùng làm baseline nhanh / ensemble member. `attn_implementation="sdpa"`, `reference_compile=False` |
| mmBERT-base (307M) | 3e-5 (thử 2e-5) | 3 | 16–32 | 256–512 | 0.06 | 0.01 | Như trên |
| **wolf-defender** (307M, init mmBERT-base) | 2e-5 (card dùng 1e-5) | 2–3 | 16–32 | 256–512 | 0.06–0.1 | 0.01 | `num_labels=3, ignore_mismatched_sizes=True`, fix RoPE nếu transformers 4.x; set `id2label/label2id` mới |
| XLM-R-base (278M) | 2e-5 | 3–4 | 32 | 256 | 0.1 | 0.01 | Ổn định, rẻ |
| XLM-R-large (560M) | **1e-5** | 3 | 16 (hoặc 8×accum 2) | 256 | **0.1** | 0.01 | grad clip 1.0; ≥2 seeds; nếu OOM bật `gradient_checkpointing_enable()`; theo dõi collapse |

Chung [R]:
- **max_len:** mean 294 chars ≈ [UL] ~70–110 tokens (tiếng Romance, tokenizer subword). → len 256 phủ phần lớn; **đo percentile 95/99 token length trên train trước**. Dùng dynamic padding (`DataCollatorWithPadding`) + sort/group by length (`group_by_length=True`) để tăng tốc mạnh. Với text dài: head+tail (vd. head 64 + tail 190 cho len 256).
- **Validation:** stratified theo `label` (và nếu được, theo `label×language`). Chọn checkpoint theo **macro-F1**, không theo loss.
- **Loss:** CE; nếu dùng class weights, dùng mức nhẹ (vd. `w_c ∝ (1/freq_c)^0.5`) rồi vẫn tune scaling trên OOF.
- **Seeds:** cố định seed; XLM-R-large chạy 2 seed.

---

## 4. Gotchas trên Kaggle

1. **transformers version** — [F] ModernBERT (mmBERT, wolf-defender) cần **transformers ≥ 4.48.0** (S2-github, S19). Model card mmBERT ghi `>=4.21.0` là sai/lỗi thời (S2) — đừng tin dòng đó. Kaggle image hiện tại dùng version nào: **chưa kiểm chứng** → `import transformers; print(transformers.__version__)` ở cell đầu; nếu < 4.48: `pip install -U "transformers>=4.48"` (cần Internet ON) hoặc attach wheel dataset.
2. **wolf-defender config v5 vs code v4** — xem mục 0.2 (RoPE theta sliding layers). Đây là bug âm thầm.
3. **bf16 không có trên T4 / P100** — [F] T4 là Turing (compute capability 7.5), không hỗ trợ bf16 (S18). → `fp16=True, bf16=False`. Không load weight ở `torch_dtype=bfloat16`. P100 (Pascal) không có tensor cores FP16 hiệu quả → [R] ưu tiên T4.
4. **Attention implementation cho ModernBERT** — [F] code ModernBERT trong 4.48: nếu user không chỉ định, nó **thử flash_attention_2 trước**, fallback SDPA → eager (S7). FlashAttention-2 yêu cầu Ampere+ (một thread báo lỗi "requires Ampere or newer" trên V100) (S20). → [R] luôn truyền `attn_implementation="sdpa"` (hoặc `"eager"` nếu gặp NaN/lỗi). [F] Có report lỗi `position_ids` khi ép SDPA bằng cách set private `_attn_implementation` trên V100 (S20) — dùng tham số public trong `from_pretrained`.
5. **`reference_compile` (torch.compile) của ModernBERT** — [F] trong 4.48, nếu `reference_compile=None` và có triton thì tự bật torch.compile; Tom Aarsen khuyên `reference_compile=False` khi crash (S7, S19). → [R] truyền `reference_compile=False` (với transformers 4.48–4.5x; ở bản mới hơn tham số có thể đã bị bỏ — nếu báo "unexpected keyword" thì bỏ đi).
6. **2×T4 + Trainer** — Trainer mặc định dùng `DataParallel` khi thấy 2 GPU trong 1 process; torch.compile không hợp với model chia device (S7). [R] Đơn giản nhất: train 2 model song song trên 2 GPU bằng 2 process (`CUDA_VISIBLE_DEVICES=0` / `=1`), hoặc chỉ dùng 1 GPU/notebook. Tránh debug DDP trong ngày deadline.
7. **fp16 NaN** — [F] có report NaN loss với ModernBERT (pretrain) và với fp16 nói chung (S21). [R] Nếu NaN: thử `attn_implementation="eager"`, giảm lr, bật grad clip, hoặc chạy fp32 cho mmBERT-small.
8. **Tokenizer** — [F] mmBERT dùng Gemma 2 tokenizer, vocab 256k, mask `<mask>`, không thêm prefix-space (S1, S2). Không trả `token_type_ids` thì OK; nếu custom model, chỉ truyền `input_ids, attention_mask` (S2-github). [F] `classifier_pooling` của mmBERT config = `"mean"` (S2-config) → `ModernBertForSequenceClassification` dùng mean pooling (mặc định class là `"cls"` nếu config không ghi) (S7-config).
9. **Dropout = 0** — [F] mmBERT/wolf config: `attention_dropout`, `mlp_dropout`, `embedding_dropout`, `classifier_dropout` đều 0.0 (S2-config, S4-config). [R] Có thể set `classifier_dropout=0.1` để regularize head (tùy chọn).
10. **Model nguồn aisuko/** — đường dẫn Kaggle Models khác HF id; kiểm tra thư mục có `config.json`, `tokenizer.json`/`tokenizer.model`. Với wolf-defender, kiểm tra config có phải bản v5 (`rope_parameters`) như trên HF không.
11. **Granite trên T4** — [F] không có `GraniteForSequenceClassification` (S8); [F] fp16 GradScaler lỗi nếu trainable params ở bf16 (S18) → cast LoRA params sang fp32. Mistral 7B: có seq-cls class, cần `pad_token` (Mistral tokenizer không có pad mặc định — [R] set `tokenizer.pad_token = tokenizer.eos_token` và `model.config.pad_token_id`).

---

## 5. Kiến thức domain (prompt injection vs jailbreak; best practice 2024–2026)

- [F] Prompt-Guard (Meta): INJECTION = instruction "out of place" (thường từ third-party data); JAILBREAK = cố override system prompt/safety; chồng lấn → ưu tiên JAILBREAK. Model card khuyên **fine-tune trên data của ứng dụng** vì attack rất application-specific; context 512, text dài thì chia segment và scan từng đoạn (S6).
- [F] Phân biệt phổ biến: prompt injection = lỗi **trust boundary** (dữ liệu không tin cậy bị coi là lệnh); jailbreak = lỗi **safety boundary** (đẩy model vượt refusal). Direct injection trông giống jailbreak; indirect injection nằm trong web/email/PDF/tool output (S22). OWASP LLM01 gộp jailbreak như một dạng prompt injection (S22) → ranh giới nhãn có thể "mờ" trong dataset.
- [F] **Over-defense** (InjecGuard/PIGuard, ACL 2025): guard model gán nhầm text benign chứa **trigger words** ("ignore", "system prompt", ...) thành attack; benchmark NotInject 339 câu benign, các guard model hiện có rơi về ~60% accuracy (S23). → [R] Làm error analysis: xem BENIGN bị predict sai thành INJECTION/JAILBREAK có chứa trigger words không; macro-F1 bị phạt nặng bởi false positive ở class PROMPT_INJECTION nhỏ.
- [F] Multilingual: GenTel-Safe dịch máy dataset jailbreak/goal-hijacking/prompt-leak sang ZH/JA/FR/ES/DE và thấy khả năng generalize của detector **khác nhau nhiều theo ngôn ngữ** (S24). Prompt-Guard: TPR 91.5%/FPR 5.3% trên multilingual jailbreak set, thấp hơn English (theo kết quả search, chưa mở card để xác minh con số) (S6, S24).
- [R] Hệ quả cho cuộc thi: (a) đánh giá macro-F1 **theo từng language** trên validation; (b) feature `language` có thể prepend vào text (vd. `"[pt] " + text`) — rẻ, thử nếu còn thời gian; (c) tail của text dài quan trọng vì payload injection hay ở cuối dữ liệu nhúng.

---

## 6. Tối ưu macro-F1 dưới imbalance

- [F] Per-class threshold là cách hiệu quả nhất cho macro-F nhưng dễ overfit label hiếm (Lin & Lin, S12). Class weight đóng vai trò như threshold đa lớp và có thể tối ưu F1 (Khim et al., S13). Logit adjustment theo label prior (post-hoc hoặc trong loss) (Menon et al., ICLR 2021, S14). Threshold tối ưu cho macro-F1 khác với threshold tối ưu accuracy (S25).
- [R] Quy trình:
  1. Lưu **OOF probs** (5-fold) hoặc holdout probs cho mỗi model + test probs.
  2. Ensemble = mean probs (có thể weight theo OOF score).
  3. Tìm `w = (w_B, w_J, w_PI)` với `w_B = 1` cố định, grid `w_J, w_PI ∈ [0.5, 4]` bước 0.1 → `pred = argmax(p * w)`; chọn theo macro-F1 OOF.
  4. Sanity check: tỷ lệ class dự đoán trên test ≈ tỷ lệ train (62/31/7) — nếu lệch nhiều thì nghi ngờ.
- [R] Class weights vs focal: với 3 class và PI 7%, CE + post-hoc scaling thường đủ; focal không bắt buộc. Không chồng quá nhiều cơ chế (weights mạnh + scaling) vì sẽ over-predict PI.
- [F] LLRD: ξ=0.95 với lr 2e-5 tốt nhất (BERT, S11). [F] R-Drop: KL hai forward pass (S15) — chỉ có ý nghĩa khi model có dropout (XLM-R có; mmBERT mặc định 0).

---

## 7. XLM-R-large: ổn định & thời gian

- [F] Instability khi fine-tune BERT/RoBERTa là do optimization (vanishing gradients), không phải forgetting; baseline khuyến nghị: lr nhỏ, warmup, train lâu hơn, bias-corrected Adam (Mosbach et al., S9). RoBERTa-based models không ổn định ở 3e-5, 1e-5 ổn cho tất cả model (S10). Có report RoBERTa-large collapse về mức random khi không warmup, thêm warmup_ratio 0.06 thì hết (S10).
- [R] Setting: lr 1e-5, warmup 0.1, linear decay, AdamW wd 0.01, grad clip 1.0, fp16, bs 16 (len 256), 3 epoch, eval mỗi ~0.5 epoch theo macro-F1. Nếu OOM: bs 8 + accum 2 + `gradient_checkpointing`.
- [UL] Thời gian: không tìm được benchmark công khai cho XLM-R-large/T4/len256 (S26 xác nhận không có). Ước lượng FLOPs: ~6×(≈300M non-embedding params)×tokens; 14k×256 tokens ≈ 3.6M tokens/epoch → ~6.5e15 FLOPs; với ~15–25 TFLOPS hiệu dụng fp16 trên T4 → **~5–10 phút/epoch nếu pad cố định 256, ít hơn với dynamic padding**, cộng overhead → để buffer 15–30 phút/epoch. **Đo thực tế 50 step đầu rồi ngoại suy.**
- [UL] Granite-2B QLoRA: ~8× compute XLM-R-large + overhead dequant 4-bit → ước tính ≥1h/epoch trên 1×T4. Mistral-7B: ≥3× Granite → nhiều giờ. Với deadline 18:00 hôm nay: **bỏ qua**.

---

## 8. Granite-3.3-2B LoRA — feasibility

- [F] Granite 3.3 2B instruct: ~2B params (Hub metadata ghi 3B), 128K context, hỗ trợ có PT/FR/IT, Apache 2.0 (S17). Architecture `GraniteForCausalLM` (S8, S27). **Không có `GraniteForSequenceClassification` trong transformers** (S8).
- [F] Nghiên cứu 2025: classifier head trên embedding token cuối của causal LLM + 4-bit LoRA vượt cách instruction-based và cạnh tranh/vượt BERT fine-tuned trên patent classification (S28) — domain khác, không chứng minh cho multilingual PI.
- [R] Nếu vẫn muốn làm (không khuyến nghị hôm nay): `AutoModel` (GraniteModel) 4-bit nf4 + `prepare_model_for_kbit_training` + LoRA r=16 trên q,k,v,o; head Linear(hidden→3) fp32 trên hidden state token cuối (padding side left hoặc lấy index `attention_mask.sum-1`); fp16 autocast, cast trainable params sang fp32 (S18); len 256, bs 8, lr 1e-4–2e-4, 1 epoch.

---

## 9. Chiến lược 6 lượt submit

[R] (private LB quyết định → tin CV hơn public LB)
1. **Sub 1 (sớm, an toàn):** wolf-defender 3-class (hoặc mmBERT-base) single model, argmax thuần. Mục đích: kiểm tra format submission + đối chiếu CV↔LB.
2. **Sub 2:** cùng model + class-prob scaling tune trên validation/OOF → đo gain của post-processing trên LB.
3. **Sub 3:** XLM-R-large single (+ scaling) → so sánh với wolf.
4. **Sub 4:** Ensemble mean probs (wolf + XLM-R-large [+ mmBERT-small/XLM-R-base]) + scaling tune trên OOF ensemble. **Ứng viên final chính.**
5. **Sub 5:** biến thể ensemble (thêm seed / bỏ model yếu / weight khác) — chỉ khi CV cải thiện rõ.
6. **Giữ 1 slot dự phòng** cho lỗi format/sự cố trước 18:00.
- Chọn final submission (nếu cuộc thi cho chọn): 1 bản ensemble tốt nhất theo **CV**, 1 bản "an toàn" (ensemble không scaling hoặc scaling nhẹ) để hedge overfit threshold.
- Không chạy theo chênh lệch public LB nhỏ (< ~0.005 macro-F1) — public LB có thể chỉ là một phần test 6,000 dòng (tỷ lệ split public/private chưa kiểm chứng).

---

## 10. Sources

- S1. mmBERT paper (arXiv 2509.06888): https://arxiv.org/html/2509.06888v1
- S2. mmBERT-base model card: https://huggingface.co/jhu-clsp/mmBERT-base ; mmBERT-small: https://huggingface.co/jhu-clsp/mmBERT-small ; S2-config: https://huggingface.co/jhu-clsp/mmBERT-base/blob/main/config.json ; S2-github (yêu cầu `transformers>=4.48.0`): https://github.com/JHU-CLSP/mmBERT ; blog: https://huggingface.co/blog/mmbert
- S3. wolf-defender model card: https://huggingface.co/patronus-studio/wolf-defender-prompt-injection
- S4-config. wolf-defender config.json: https://huggingface.co/patronus-studio/wolf-defender-prompt-injection/resolve/main/config.json
- S5. protectai deberta-v3-base-prompt-injection-v2: https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2
- S6. Meta Prompt-Guard-86M model card: https://huggingface.co/meta-llama/Prompt-Guard-86M
- S7. transformers v4.48.0 ModernBERT code: https://github.com/huggingface/transformers/blob/v4.48.0/src/transformers/models/modernbert/modeling_modernbert.py ; S7-config: https://github.com/huggingface/transformers/blob/v4.48.0/src/transformers/models/modernbert/configuration_modernbert.py
- S8. transformers auto mapping (main): https://github.com/huggingface/transformers/blob/main/src/transformers/models/auto/modeling_auto.py ; Granite docs: https://huggingface.co/docs/transformers/model_doc/granite
- S9. Mosbach et al., On the Stability of Fine-tuning BERT: https://arxiv.org/abs/2006.04884
- S10. HF forum RoBERTa instability: https://discuss.huggingface.co/t/does-anyone-else-observer-roberta-fine-tuning-instability/6499 ; XLM-R tuning thread: https://discuss.huggingface.co/t/fine-tune-xlm-roberta-large-xnli/7502 ; collapse w/o warmup report: https://arxiv.org/pdf/2604.16787
- S11. Sun et al., How to Fine-Tune BERT for Text Classification? (head+tail, LLRD): https://arxiv.org/abs/1905.05583
- S12. Lin & Lin, thresholding / macro-F: https://www.csie.ntu.edu.tw/~cjlin/papers/thresholding/smooth_acm.pdf
- S13. Khim et al., class weights & F1: https://arxiv.org/abs/2004.04715
- S14. Menon et al., Long-tail learning via logit adjustment: https://arxiv.org/abs/2007.07314
- S15. R-Drop: https://arxiv.org/abs/2106.14448
- S16. Focal loss (Lin et al. 2017): https://arxiv.org/abs/1708.02002
- S17. Granite-3.3-2b-instruct card: https://huggingface.co/ibm-granite/granite-3.3-2b-instruct
- S18. T4 không hỗ trợ bf16 / fp16 GradScaler issue: https://discuss.pytorch.org/t/current-cuda-device-does-not-support-bfloat16-please-switch-dtype-to-float16/201564 ; https://github.com/unslothai/unsloth/issues/4082
- S19. ModernBERT cần transformers 4.48 + `reference_compile=False`: https://huggingface.co/answerdotai/ModernBERT-base/discussions/3
- S20. ModernBERT FA2 cần Ampere / lỗi SDPA trên V100: https://huggingface.co/answerdotai/ModernBERT-base/discussions/56
- S21. ModernBERT NaN loss thread: https://discuss.huggingface.co/t/modernbert-maskedlm-nan-training-loss/133951
- S22. Prompt injection vs jailbreaking: https://promptfoo.dev/blog/jailbreaking-vs-prompt-injection/ ; https://www.pillar.security/blog/the-terminology-problem-causing-security-teams-real-risks
- S23. InjecGuard / PIGuard (over-defense, NotInject): https://arxiv.org/abs/2410.22770v1 ; https://preview.aclanthology.org/setup/2025.acl-long.1468 ; https://huggingface.co/leolee99/PIGuard
- S24. GenTel-Safe (multilingual PI benchmark): https://arxiv.org/pdf/2409.19521 ; Sentra-Guard: https://arxiv.org/html/2510.22628v1
- S25. Multiclass threshold-based classification: https://arxiv.org/pdf/2505.11276
- S26. (Không có benchmark XLM-R-large/T4 công khai — kết quả search tiêu cực) ; RoBERTa-large tutorial A100: https://arxiv.org/pdf/2403.02504
- S27. NeMo AutoModel Granite coverage: https://docs.nvidia.com/nemo/automodel/latest/model-coverage/llm/ibm/granite.html
- S28. Fine-Tuning Causal LLMs for Text Classification (embedding vs instruction): https://arxiv.org/html/2512.12677v1

### Những gì CHƯA kiểm chứng
- Version transformers/torch mặc định của Kaggle image hiện tại; flash-attn có cài sẵn không.
- Thời gian train thực tế trên T4 (mọi con số thời gian là [UL]).
- Token length thực tế của data (ước lượng 70–110 tokens trung bình là [UL]).
- Convention gán nhãn của dataset cuộc thi (giả định giống Prompt-Guard).
- Hành vi của transformers 5.x khi load config mmBERT gốc (key `global_rope_theta/local_rope_theta` cũ).
- Con số Prompt-Guard TPR 91.5%/FPR 5.3% multilingual (lấy từ tóm tắt search, chưa đọc trực tiếp trên card).
