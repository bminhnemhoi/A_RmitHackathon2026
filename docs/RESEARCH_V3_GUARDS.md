# RESEARCH V3: Guard models và các kỹ thuật chuyển giao được (2026-10-22, ~14:15 ICT)

Bối cảnh: OOF macro-F1 0.977 (ensemble). Top public LB 0.9906. Lỗi còn lại gần như toàn bộ là **pt BENIGN↔JAILBREAK** trên prompt role-play dài kiểu WildJailbreak (adversarial-benign vs adversarial-harmful). Mistral-7B LoRA (1 epoch, last-token head, r16, len256) cho fold 0.964.
Các guard model dưới đây **không được dùng** làm backbone hay teacher. Tài liệu này chỉ lấy *kỹ thuật*.
Nhãn: **[V]** đã đọc trong nguồn gốc (paper/model card). **[S]** nguồn thứ cấp (review/aggregator). **[U]** suy luận của mình, chưa kiểm chứng.

---

## (a) Bảng guard models

| Model | Base | Training data | Label scheme | Multilingual | Recipe đáng chú ý | Điểm báo cáo |
|---|---|---|---|---|---|---|
| **Llama Prompt Guard 2 86M / 22M** ([HF card](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M), [Meta docs](https://developer.meta.com/ai/docs/model-cards-and-prompt-formats/prompt-guard/)) | mDeBERTa-v3-base (86M), DeBERTa-v3-xsmall (22M) | benign web data, user prompts, LLM instructions, public injection+jailbreak sets, synthetic injections, red-team data [V] | binary benign / malicious (bỏ nhãn "injection" của PG1) [V] | 86M: en, fr, de, hi, it, **pt**, es, th [V] | **CE + energy penalty**: "penalty for large negative energy predictions on benign prompts" (theo Liu et al. 2020, energy-based OOD) [V]. Tokenizer sửa để chống adversarial tokenization (whitespace, fragmented tokens) [V]. ctx 512 | AUC en .998, multilingual .995. Recall@1%FPR 97.5% (PG1: 21.2%), tác giả quy gain này cho custom loss [V] |
| **Qwen3Guard-Gen / Stream** 0.6/4/8B ([arXiv 2510.14276](https://arxiv.org/abs/2510.14276)) | Qwen3 instruct (Gen), Qwen3 base + 2 heads (Stream) | 1.19M mẫu, human + synthetic (Self-Instruct từ taxonomy), pre-label bằng nhiều Qwen model rồi vote [V/S] | Safe / **Controversial** / Unsafe + category, có "Jailbreak (input only)" [V] | 119 ngôn ngữ; **dịch sang 15 ngôn ngữ bằng Qwen-MT**, lọc bằng language-mixing detection, LLM judge và manual review [V] | vanilla SFT. Hai data split với tỉ lệ Safe/Unsafe khác nhau ("strict" vs "loose"), mẫu bất đồng thì gán Controversial. Tăng gấp đôi Safe làm model "more permissive". Distill từ Qwen3-32B: +0.5 đến +1.1 [V] | RTP-LX prompt 85.0, en prompt avg 90.0 (8B strict) [V]. Bộ attack mới chỉ đạt 33.8% ([2511.22047](https://arxiv.org/html/2511.22047v1)) |
| **WildGuard** 7B ([arXiv 2406.18495](https://arxiv.org/abs/2406.18495), [HF](https://huggingface.co/allenai/wildguard)) | Mistral-7B-v0.3, **full FT** | WildGuardMix 86,759 mẫu: adversarial harmful 4,489+11,289, **adversarial benign 4,339+11,411**, vanilla, in-the-wild, annotator-written. Train do GPT-4 gán nhãn, test do người gán [V] | 3 output text: harmful request / refusal / harmful response [V] | chỉ en | **Instruction prompt + output nhãn dạng text**. 2 epochs, lr 2e-6, linear schedule, warmup 0.03, wd 0, bs 128, len 4096 [V]. Đổi base sang Llama3/Gemma/OLMo thì kết quả gần như không đổi, data quan trọng hơn base [V] | Prompt F1: adversarial 85.5, vanilla 91.7. **Bỏ synthetic adversarial data thì adversarial F1 giảm 8.4** [V] |
| **WildJailbreak** dataset ([arXiv 2406.18510](https://arxiv.org/abs/2406.18510)) | n/a (data) | 262K: vanilla/adversarial × harmful/benign. Adversarial benign = prompt benign được bọc tactic jailbreak (GPT-4 / Mixtral) rồi lọc low-risk [V] | harmful / benign × vanilla / adversarial | en | contrastive: cùng tactic wrapper cho cả 2 lớp, chỉ khác intent gốc [V] | "all components contribute" [V] |
| **PolyGuard** ([arXiv 2504.04377](https://arxiv.org/abs/2504.04377), [data](https://huggingface.co/datasets/ToxicityPrompts/PolyGuardMix)) | Qwen2.5-7B-Instruct, Ministral-8B, Qwen2.5-0.5B, **LoRA** | PolyGuardMix 1.91M, 17 ngôn ngữ (**có pt, fr, it**): **bản dịch WildGuardMix** (TowerInstruct / NLLB [S]) + chat đa ngôn ngữ thật [V] | giống WildGuard (3 boolean) + MLCommons category | 17 ngôn ngữ | LoRA SFT generative [V] | +5.5% so với SOTA (tự báo cáo) [V] |
| **Granite Guardian 3.x** ([arXiv 2412.07724](https://arxiv.org/abs/2412.07724)) | Granite 3.x 2B/8B (3.2: 5B) | human annotation + **synthetic adversarial jailbreak** (revision-based transformation), sub-sample rồi gán nhãn [V/S] | Yes/No + risk name trong prompt | chủ yếu en | dạng Yes/No token; "adversarial intent" được tính là một risk riêng [S] | 3.2-5B generalize tốt nhất (gap 6.5%) theo [2511.22047](https://arxiv.org/html/2511.22047v1) |
| **ShieldGemma** 2/9/27B ([card](https://ai.google.dev/gemma/docs/shieldgemma/model_card)) | Gemma 2 | synthetic + human, 4 harm type | **P(Yes)/P(No) token logits** | en | label-token scoring [V] | 27B F1 0.830 (internal) [V] |
| **Llama Guard 3/4** | Llama 3.1-8B / Llama 4 Scout (12B) | MLCommons hazards | text "safe/unsafe + S-codes" | LG3: 8 ngôn ngữ (có pt, fr, it) [S] | SFT, first-token prob làm score | n/a |
| **NemoGuard JailbreakDetect** ([card](https://huggingface.co/nvidia/NemoGuard-JailbreakDetect), [NIM](https://docs.nvidia.com/nim/nemoguard-jailbreakdetect/latest/index.html)) | **Random forest trên embedding** snowflake-arctic-embed-m-long (768-d) | open datasets (có WildJailbreak), augment bằng garak [V] | bool + prob | en | embedding + classical classifier [V] | JailbreakHub F1 0.960, FPR 0.4% [S] |
| **PIGuard (InjecGuard)** ([ACL 2025](https://aclanthology.org/2025.acl-long.1468), [HF](https://huggingface.co/leolee99/PIGuard)) | DeBERTa-v3-base | public PI data + **NotInject** (339 câu benign chứa trigger word) [V] | binary | en | **MOF**: tìm token gây over-defense (token-wise recheck), sinh câu benign chứa token đó rồi train lại [V] | +30.4% trên NotInject so với best trước đó [V] |
| **Sentinel** (Qualifire) ([arXiv 2506.05446](https://arxiv.org/abs/2506.05446)) | ModernBERT-large | public mix có WildJailbreak [V] | binary | en | standard FT encoder | F1 0.980 internal [V] |

Ghi chú cộng đồng:
- Không tìm thấy thread Reddit r/LocalLLaMA, X, Zhihu hay CSDN có nội dung kỹ thuật cụ thể trong time-box.
- Hai model card cộng đồng đáng tham khảo. Một DeBERTa fine-tune có F1 thấp nhất ở slice **role_play (0.964)** ([alvi42](https://huggingface.co/alvi42/prompt-injection-guard-v1)), tức đúng failure mode của mình. Dự án [bert-tiny-injection-detector](https://huggingface.co/marklkelly/bert-tiny-injection-detector) loại WildJailbreak vì trộn vào làm recall giảm khoảng 20 điểm: **mapping nhãn và tỉ lệ trộn phải kiểm chứng**.
- Một model XLM-R zh/en ([Unified_Prompt_Guard](https://huggingface.co/ynyg/Unified_Prompt_Guard)) dùng back-translation và paraphrase augmentation.
- Paper so sánh head và label-token ([arXiv 2512.12677](https://arxiv.org/abs/2512.12677)): với single-label, **last-token head ≥ instruction/label-token**. Instruction format chỉ cạnh tranh ở multi-label với budget lớn.

---

## (b) Top-6 cải tiến chuyển giao được, xếp theo expected gain cho setup của mình

Ngân sách: khoảng $5 Modal. H100 khoảng $3.95/h, L40S khoảng $1.95/h [U, giá xấp xỉ]. Thời gian còn khoảng 3.5h.

### 1. External contrastive data cùng nguồn: WildJailbreak / PolyGuardMix-pt adversarial benign + harmful, train lại encoder tốt nhất
- **Bằng chứng:** WildGuard ablation: bỏ adversarial synthetic data thì adversarial F1 giảm 8.4 [V]. WildJailbreak được thiết kế contrastive (cùng wrapper, khác intent) [V]. Đây chính là confusion của mình.
- **Recipe:**
  - Lấy `allenai/wildjailbreak` train split, gồm `adversarial_benign` và `adversarial_harmful` (license ODC-BY, gated).
  - Hoặc lấy PolyGuardMix split **pt** (có sẵn bản dịch WildGuardMix sang pt, fr, it), lọc `adversarial==True`.
  - Lấy 3k–5k mỗi lớp. Map harmful → JAILBREAK, benign → BENIGN.
  - Chỉ thêm vào **train fold**, không thêm vào validation. Sample weight 0.5 cho external.
  - mmBERT-base / XLM-R-large, recipe hiện tại (lr 3e-5, 3–5 epochs, len 384).
- **Rủi ro:**
  - (i) Rule external data: CONTEXT.md không cấm, nhưng **phải kiểm tra trang Rules / Kaggle discussion**. Data phải public và free.
  - (ii) Mapping nhãn có thể lệch định nghĩa của BTC, ví dụ BTC có thể gán vanilla harmful là BENIGN. Kiểm tra nhanh trước: grep vài prompt train pt dài, so với WildJailbreak en để xem có cùng nguồn không. Nếu có cùng nguồn thì mapping gần như chắc đúng.
  - (iii) Dùng bản dịch PolyGuard là dùng output của MT model ngoài danh sách. Đây không phải backbone nên gray zone, ghi rõ trong notebook.
- **Gain kỳ vọng:** +0.003–0.010 OOF trên pt [U]. **Chi phí:** khoảng $0.3–0.5/fold trên L40S. 5 folds khoảng $2.

### 2. Uncertainty-gated LLM re-scoring, dùng Mistral OOF đã có (gần như $0)
- **Bằng chứng:**
  - WildGuard cho thấy 7B có prior về harmfulness [V].
  - Mistral fold 0.964 thấp hơn encoder, nhưng lỗi có thể ít tương quan [U].
  - Kiểu cascade (guard nhỏ trước, LLM cho ca khó) là pattern chung của production guardrails [S].
- **Recipe:**
  - Trên fold đã có Mistral OOF, lấy tập mẫu mà ensemble có margin |p_B − p_J| < 0.3.
  - So accuracy của Mistral và ensemble trên tập này.
  - Nếu Mistral tốt hơn: dùng `p = α·p_mistral + (1−α)·p_ens` chỉ trong vùng uncertain. Chọn α ∈ {0.3, 0.5, 0.7} bằng nested CV.
- **Gain kỳ vọng:** 0 đến +0.004 [U]. **Chi phí:** chỉ analysis CPU. Cần thêm Mistral inference trên test khoảng 6k mẫu (len 512, khoảng $0.3 H100).

### 3. Mistral LoRA v2: 2 epochs, len 512 head+tail, r32, instruction prompt (giữ head)
- **Bằng chứng:**
  - WildGuard và PolyGuard đều dùng **2 epochs** với instruction format [V].
  - 1 epoch ở r16 thường under-fit trên LoRA classifier [U].
  - len256 cắt mất phần payload (intent) ở cuối các prompt role-play dài: cần kiểm tra % pt-JAILBREAK lỗi có > 256 token [U].
  - Head trên last token tương đương hoặc tốt hơn label-token cho single-label ([2512.12677](https://arxiv.org/abs/2512.12677)), nên **giữ seq-cls head**. Chỉ bọc text trong instruction ngắn: "Classify the user prompt as BENIGN, JAILBREAK or PROMPT_INJECTION. Prompt: {text}".
- **Recipe:** r 32, alpha 64, dropout 0.05, target all linear (q,k,v,o,gate,up,down). LoRA lr 1e-4, head lr 2e-4, cosine, warmup 5%. 2 epochs, bs 16. Truncation head 384 + tail 128. bf16, grad ckpt. Class weight nhẹ (B 1.0, J 1.2, PI 1.5).
- **Gain kỳ vọng:** member Mistral +0.005–0.01 đơn lẻ. Ensemble +0.001–0.004 [U].
- **Chi phí:** khoảng 30–45 phút/fold trên H100 ($2–3) [U]. **Chỉ đủ tiền 1 fold + full-train**, nên ưu tiên #1 trước.

### 4. Prompt Guard 2 energy margin hoặc focal loss chỉ cho cặp B↔J
- **Bằng chứng:** PG2: CE + energy penalty cho benign tăng Recall@1%FPR từ 21% lên 97.5% [V]. Gain đó chủ yếu về calibration ở low-FPR, **không trực tiếp là macro-F1** [U].
- **Recipe (encoder):** `L = CE + λ·[relu(E_benign − m_in)^2 + relu(m_out − E_attack)^2]`, với `E = −logsumexp(logits)`, λ 0.1, m_in −7, m_out −3 (theo Liu et al. 2020, [arXiv 2010.03759](https://arxiv.org/abs/2010.03759)). Hoặc dùng focal γ=2.
- **Gain kỳ vọng:** 0 đến +0.002 [U]. **Chi phí:** khoảng $0.4/fold. Ưu tiên thấp.

### 5. MOF-style trigger-word debiasing cho role-play pt
- **Bằng chứng:** PIGuard/MOF giảm over-defense bằng cách sinh benign chứa trigger token [V].
- **Analog:** tìm n-gram pt (ví dụ "imagine que", "você é um personagem", "no mundo fictício") có tỷ lệ JAILBREAK cao trên OOF-errors. Đảm bảo có adversarial-benign chứa các n-gram này: lấy từ #1, hoặc sinh bằng Mistral (allowed backbone) khoảng 500 câu.
- **Gain kỳ vọng:** +0–0.003 [U]. **Chi phí:** generation khoảng $0.5 + retrain. Chỉ làm nếu #1 bị cấm.

### 6. Seed / epoch ensembling của encoder tốt nhất trên data đã tăng cường
- Đây là kỹ thuật generic: 2–3 seeds của XLM-R-large / mmBERT-base.
- **Gain kỳ vọng:** +0.001–0.003 [U]. **Chi phí:** khoảng $0.3/seed-fold trên L40S.

**Không khuyến nghị:**
- Chuyển sang label-token logits thay head: [2512.12677](https://arxiv.org/abs/2512.12677) không ủng hộ.
- CoT / generative reasoning (chậm).
- Contrastive SupCon: không tìm thấy bằng chứng trực tiếp cho guard [U].
- Per-class thresholds: đã fail nested CV (RESEARCH_V2).

**Thứ tự đề xuất trong 3.5h, $5:**
1. 10 phút: kiểm tra rule và xác nhận dataset trùng nguồn WildJailbreak.
2. #2 (analysis, $0).
3. #1 với 5 folds mmBERT-base hoặc XLM-R-large (khoảng $2).
4. Nếu còn khoảng $3 thì làm #3 full-train để predict test, dùng α từ fold cũ.

---

## (c) Giả thuyết "top teams đã làm gì" (0.9906) [U, toàn bộ]
1. **Nhận ra data là bản dịch của WildJailbreak / WildGuardMix** (và các PI set như deepset/hackaprompt). Train thêm bằng data gốc hoặc bản dịch (PolyGuardMix có sẵn pt/fr/it). Đây là đòn bẩy lớn nhất cho đúng confusion adversarial benign↔harmful.
   - Biến thể gray/không nên làm: tra cứu nearest-neighbour test → external label. Gần với leakage, cần đọc rule.
2. **LLM 7B LoRA train kỹ hơn** (2–3 epochs, len lớn, nhiều fold), làm member chính hoặc re-scorer trên vùng uncertain. Kết hợp encoder ensemble.
3. **Pseudo-label test** với ngưỡng cao (≥0.95), sau đó retrain.
4. Ít khả năng: thủ thuật loss đặc biệt (energy, focal). Gain chính đến từ data và model capacity, khớp với kết luận WildGuard "data > base model".

## Nguồn
- Prompt Guard 2: https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M , https://developer.meta.com/ai/docs/model-cards-and-prompt-formats/prompt-guard/
- Qwen3Guard: https://arxiv.org/abs/2510.14276 , review https://www.themoonlight.io/en/review/qwen3guard-technical-report , https://alphaxiv.org/overview/2510.14276v1
- WildGuard: https://arxiv.org/abs/2406.18495 (Appendix B recipe, Table 5 ablation)
- WildJailbreak / WildTeaming: https://arxiv.org/abs/2406.18510
- PolyGuard: https://arxiv.org/abs/2504.04377 , https://huggingface.co/ToxicityPrompts/PolyGuard-Qwen
- Granite Guardian: https://arxiv.org/abs/2412.07724
- ShieldGemma: https://ai.google.dev/gemma/docs/shieldgemma/model_card
- NemoGuard: https://huggingface.co/nvidia/NemoGuard-JailbreakDetect
- PIGuard / InjecGuard: https://aclanthology.org/2025.acl-long.1468 , https://huggingface.co/leolee99/PIGuard
- Sentinel: https://arxiv.org/abs/2506.05446
- Guardrail robustness eval: https://arxiv.org/html/2511.22047v1
- Head vs instruction classification: https://arxiv.org/abs/2512.12677
- Multilingual jailbreak detection (MLJailDe): https://arxiv.org/pdf/2606.11202
- Community model cards: https://huggingface.co/alvi42/prompt-injection-guard-v1 , https://huggingface.co/marklkelly/bert-tiny-injection-detector , https://huggingface.co/ynyg/Unified_Prompt_Guard
- Guardrail blog (FP compounding): https://blog.premai.io/production-llm-guardrails-nemo-guardrails-ai-llama-guard-compared
