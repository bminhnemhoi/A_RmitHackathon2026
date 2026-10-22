# RESEARCH V4 (scout): "Jev TypeSafe", tài liệu 2024–2026 và các phương pháp hợp lệ cho bottleneck pt BENIGN↔JAILBREAK

Ngày 2026-10-22, khoảng 15:25–16:00 ICT. Vai trò: research scout (chỉ CPU, không chạy Modal/GPU).
Ràng buộc đã xác nhận: chỉ dùng backbone được duyệt; chỉ train trên `train.csv`; **không external data**; không gán nhãn test bằng tay hay bằng LLM. Pseudo-label từ model của mình trên test text thì team lead đã chấp nhận.
Script CPU của tài liệu này: `C:\Users\Admin\AppData\Local\Temp\claude\d--A-RMITHACKATHON\1170d8c9-19f8-4773-beaf-7cf8b7cc5e2a\scratchpad\team\research-scout\margin_cascade.py` và `lens.py` trong cùng thư mục.

Quy ước nhãn độ tin cậy:
- **[V]**: đã đọc trực tiếp trang abstract/HTML trong phiên này.
- **[S]**: chỉ dựa vào tóm tắt của search hoặc nguồn thứ cấp (blog, review), chưa đọc bản gốc.
- **[M]**: link và nội dung lấy từ trí nhớ, không mở lại trong phiên này.
- **[U]**: suy luận hoặc ước lượng của mình, chưa kiểm chứng.
- **[D]**: đo trên data của mình (OOF), có script.

---

## 0. TL;DR

1. **"Jev Tysafe" gần như chắc chắn là Jev của TypeSafe AI** [S/V]. Đây là một "System One model" độc quyền, chạy qua hosted API. Nó không sinh text mà trả về **typed decisions** (Choice / Score / Noul) kèm xác suất đã calibrate, và có pattern "act khi chắc, chuyển tiếp (route) khi không chắc".
   - **Không được dùng Jev**: nó không nằm trong danh sách backbone, và gửi text cuộc thi tới API ngoài là vi phạm.
   - **Học được cơ chế quyết định (decision mechanism) của nó**: confidence bands, margin, cascade System One → System Two, và option có mô tả (label descriptions).
2. **Đo trên OOF của blend9 [D]** (đây là "Jev-style band analysis"):
   - 86% lỗi (246/286) nằm ở margin < 0.95, tức nhóm 1,503 dòng (10.7%).
   - 54.5% lỗi nằm ở margin < 0.5, tức chỉ 462 dòng (3.3%). Test có 191 dòng ở band này.
   - 40 lỗi ở margin ≥ 0.95 gần như là sàn label-noise.
   - ⇒ Hướng đúng là một **System Two mạnh hơn cho khoảng 3–10% mẫu có margin thấp**, không phải chỉnh threshold.
3. **Mistral fold-0 hiện có (1 epoch, r16, len256) KHÔNG đủ làm System Two [D]**. Trên band margin < 0.3, accuracy của nó là 0.71 so với blend 0.67 (n=52, nhiễu). Trộn có gate cho fold0 nhiều nhất +0.0008, nằm trong nhiễu và được chọn in-sample.
   - Lý do khả dĩ [D/U]: **43% lỗi pt dài hơn 850 chars**, mà len256 token Mistral chỉ đọc được khoảng 850 chars (pt ~3.3 chars/token, [U]). Như vậy Mistral bị cắt đúng ở phần khó.
4. **Đề xuất mới #1 (rẻ, hợp lệ): linear probe trên hidden states của Mistral-7B-Instruct-v0.3 (frozen)**.
   - Bằng chứng: PIShield, JBShield (USENIX Sec'25), Constitutional Classifiers++ (Anthropic, 2026, probe + cascade), và paper "harmful intent recoverable from residual stream".
   - GPU chỉ là một lần forward cho 20k text ở len 1024 (khoảng 7–10 phút H100, ~$0.5–0.8 [U]). Phần còn lại (LR theo layer, 5-fold) chạy CPU.
   - Kết quả cho ra OOF và test probs của một member 7B có độ đa dạng (diversity) cao.
5. **Đề xuất #2**: Mistral LoRA v2 làm System Two: len 1024 (head+tail), 2 epochs, r32, prompt có **mô tả nhãn** (học từ Jev option descriptions).
6. **Đề xuất #3**: ensemble self-distillation (born-again) từ blend9 soft-OOF sang mmBERT-base/wolf. **Đề xuất #4**: label smoothing 0.05–0.1 và bỏ khoảng 40 mẫu "chắc chắn sai nhãn" khi refit cuối. Gain kỳ vọng nhỏ.
7. Lý do có thể khiến top LB đạt 0.99 [U]: (a) external data cùng nguồn WildJailbreak (với mình là bị cấm); (b) LLM 7B fine-tune đúng cách với context đủ dài; (c) cả hai. Với ràng buộc của mình, (b) là đòn bẩy duy nhất còn lại có quy mô lớn.

---

## 1. "Decision của Jev Tysafe" là gì

### 1.1 Nhận diện
- Search các biến thể "TySafe", "Tysafe", "JEV safe" không ra gì liên quan. "TyreSafe" là một hội từ thiện về lốp xe ở UK, không liên quan.
- Search "JEV safe LLM guard decision" thì ra **Jev của TypeSafe AI**. Mình kết luận "Jev Tysafe" = **Jev / TypeSafe** (gõ sai chính tả). Đây là **suy luận [U] có độ tin cậy cao**: tên trùng và chủ đề "decision" trùng. Nếu user muốn nói thứ khác thì cần hỏi lại.
- Nguồn:
  - Datadog Security Labs, "Jev as a judge: how counterfeit policy steers Jev" [V]: https://securitylabs.datadoghq.com/articles/jev-as-a-judge-how-counterfeit-policy-steers-jev.md
  - Wu & Lim, "Decision Hijacking: Prompt Injection Attacks on Jev's Typed Probabilistic Decisions", arXiv 2609.28613, 23/09/2026 [V]: https://arxiv.org/abs/2609.28613 , https://arxiv.org/html/2609.28613v1
  - Victor Dibia newsletter, "How Jev works: calibrated decision…" [V, đây là bài phân tích của bên thứ ba]: https://newsletter.victordibia.com/p/how-jev-works-calibrated-decision
  - noze.it, "Jev and System One models: typed decisions" [V, bên thứ ba]: https://www.noze.it/en/insights/jev-system-one-typed-decisions/
  - DataCamp blog [S]: https://www.datacamp.com/blog/system-one-models-jev

### 1.2 Cơ chế (vendor claim, phần lớn chưa được công bố)
- **Định nghĩa**: "unstructured state in, typed probabilistic decisions out". Jev không sinh string và trả lời nhiều câu hỏi song song [V, vendor quote theo Dibia/Datadog].
- **3 primitive** [V, Datadog/noze]:
  - **Choice**: tối đa 255 option, mỗi option có mô tả do developer viết. Trả về `choice`, `probabilities`, `confidence`.
  - **Score**: thang có thứ tự 2–10 mức.
  - **Noul**: Bernoulli, trả về một xác suất.
- **Calibration**: TypeSafe tuyên bố huấn luyện bằng "RLCD, Reinforcement Learning for Calibrated Decisions", tức "honest probabilities". Phương pháp chưa công bố, nên **không kiểm chứng được** [V: các bài review nói rõ điểm này].
- **Option có mô tả giúp tăng accuracy**: một bên thứ ba đo trên 662 prompt-injection message (deepset). Có task description: acc 96.5%, ECE 0.059. Không có: 89.7%, ECE 0.093 [V, số do bên thứ ba đo, được trích trong bài Dibia].
- **Decision pattern được khuyến nghị** [V, noze/Dibia]:
  - Chia 3 confidence band: cao thì act, trung bình thì xác nhận, thấp thì route sang người hoặc model khác.
  - Đặt threshold dựa trên chi phí lỗi.
  - Đo coverage.
  - Có option "none/other" để abstain.
- **Decision margin** [V, Wu & Lim]: định nghĩa margin = p(đúng) − p(đối thủ). Hijack thành công 12/12 khi margin ≤ 0.1, nhưng chỉ 1/468 khi margin xa biên. Như vậy lỗi tập trung ở margin nhỏ.
- **Hiệu năng PI của Jev** [V, Datadog], trên 12,295 mẫu (deepset, xTRam1, zachz, yanismiraoui):
  - Noul: P 99.66%, R 82.90%, F1 90.51%.
  - Choice: F1 91.93%.
  - Khi chèn "counterfeit policy" vào input, recall sụp xuống 3.9% / 9.7%.
  - ⇒ Jev **không mạnh hơn** các fine-tuned encoder chuyên biệt về PI. Giá trị nằm ở *cách ra quyết định*, không nằm ở độ chính xác.

### 1.3 Áp dụng vào bài của mình (hợp lệ)

| Cơ chế Jev | Phiên bản hợp lệ cho mình | Trạng thái |
|---|---|---|
| Calibrated probabilities | Ensemble đã calibrate tốt (ECE 0.006, ERRORS.md). Temperature scaling không đổi argmax của từng model, nên gain kỳ vọng ~0 | Không cần làm thêm |
| Confidence bands + routing sang System Two | Cascade: blend9 quyết định khi margin ≥ τ, còn lại chuyển sang model mạnh hơn (Mistral probe/LoRA v2). τ chọn bằng nested CV | **Đã đo band [D], xem 1.4**. Cần System Two đủ mạnh |
| Option descriptions (task description) | Trong prompt Mistral LoRA/probe, ghi định nghĩa 3 nhãn bằng tiếng Anh ngắn: BENIGN = yêu cầu bình thường, kể cả fiction/role-play có ý định vô hại; JAILBREAK = cố vượt chính sách để lấy nội dung có hại; PROMPT_INJECTION = chèn lệnh ghi đè chỉ thị hệ thống | Chưa thử. Chi phí ~0 nếu chạy kèm #2 |
| "Score independently then choose" (2 stage) | Hierarchical: head 1 là attack vs benign, head 2 là J vs PI; hoặc một specialist B-vs-J cho mẫu dài | Ưu tiên thấp. Lỗi pt là B↔J đối xứng 109/104, nên specialist chỉ có ích nếu nó *giỏi hơn*, không phải vì bias |
| Margin là chỉ báo lỗi | Dùng margin để chọn mẫu test cho pseudo-label (chỉ lấy margin ≥ 0.95), và để gate System Two | Pseudo-label đã dùng. Gating cần System Two mới |
| Abstain / "none" | Không áp dụng được: metric bắt buộc phải gán nhãn cho mọi dòng | n/a |

### 1.4 Band analysis trên blend9 OOF [D] (`margin_cascade.py`)

| margin band | n train | errors | err% | cum % errors | n test |
|---|---:|---:|---:|---:|---:|
| [0, 0.1) | 88 | 35 | 39.8 | 12.2 | 34 |
| [0.1, 0.2) | 93 | 39 | 41.9 | 25.9 | 26 |
| [0.2, 0.3) | 85 | 28 | 32.9 | 35.7 | 46 |
| [0.3, 0.5) | 196 | 54 | 27.6 | 54.5 | 85 |
| [0.5, 0.8) | 581 | 68 | 11.7 | 78.3 | 246 |
| [0.8, 0.95) | 460 | 22 | 4.8 | 86.0 | 334 |
| [0.95, 1] | 12,497 | 40 | 0.32 | 100 | 5,229 |

- blend9 OOF có 286 errors, macro-F1 0.9770.
- Ở pt: 79/219 lỗi có margin < 0.3.
- Nếu System Two đạt **acc 85%** trên band < 0.5 (462 dòng), số lỗi giảm từ 156 xuống khoảng 69. Như vậy OOF macro-F1 lên khoảng **0.984** [U, ngoại suy tuyến tính].
- Mistral fold-0 hiện có chỉ đạt khoảng 0.72 trên band đó, ngang blend.
- Tính khả thi:
  - **Cần một System Two khác về chất**: context đủ dài, capacity 7B, nhìn được toàn bộ "core request".
  - Band 0.5–0.8 còn 68 lỗi, nên τ tối ưu có thể nằm quanh 0.8, tức route khoảng 10% dữ liệu.

---

## 2. Vì sao top LB (0.9906) có thể làm tốt như vậy [U, toàn bộ là giả thuyết]

- Public LB có khoảng 1,800 dòng. Macro-F1 0.9906 tương ứng khoảng 15–20 lỗi, tức error rate ~1%. OOF của mình ~2.0% (286/14k). Sàn noise ước lượng ~0.3% (40 lỗi ở margin ≥ 0.95).
- ⇒ Top teams **không** vượt sàn noise. Họ sửa được khoảng một nửa số lỗi "học được" ở pt B↔J.
- Ba con đường khả dĩ:
  1. **External data cùng nguồn** (WildJailbreak gốc tiếng Anh, PolyGuardMix có bản dịch pt). WildGuard ablation cho thấy bỏ adversarial synthetic data thì adversarial F1 giảm 8.4 điểm ([2406.18495](https://arxiv.org/abs/2406.18495) [M/V trong RESEARCH_V3]). **BTC đã cấm. Mình không làm.**
  2. **LLM 7B fine-tune đúng cách** (Mistral-7B-Instruct-v0.3 có trong danh sách). WildGuard chính là **Mistral-7B-v0.3 full FT** cho đúng bài adversarial-benign vs adversarial-harmful. Tác giả kết luận "data > base model" [V trong V3]. Ở các cuộc thi Kaggle 2024–2025 (LMSYS, WSDM Cup), lời giải top đều là LLM LoRA 7–9B cộng distillation [S: [nebius review](https://nebius.com/blog/posts/chatbot-arena-competition-review)].
  3. **Pseudo-label + distillation nhiều vòng** trên test text. Đây là kỹ thuật chuẩn của Kaggle và hợp lệ với mình.
- Mình không thấy dấu hiệu leak thứ tự id (đã kiểm tra).

---

## 3. Khảo sát tài liệu 2024–2026 (chọn lọc theo mức liên quan tới bottleneck)

### 3.1 Detector dựa trên hidden states (đọc nội tại LLM), liên quan nhất tới đề xuất #1
| Work | Venue / ngày | Ý chính | Tag |
|---|---|---|---|
| PIShield, Zou et al. | arXiv [2510.14005](https://arxiv.org/abs/2510.14005) (10/2025, v3 01/2026) | Logistic regression trên residual stream của **last token**, chọn "injection-critical layer" theo validation. Không fine-tune LLM. FPR/FNR thấp trên 5 benchmark, vượt 11 baseline | [V abstract]. Có tái hiện độc lập: kém khi chỉ có 200 mẫu, tốt khi có full data ([2608.22248](https://arxiv.org/pdf/2608.22248) [S]) |
| Harmful Intent as a Geometrically Recoverable Feature of LLM Residual Streams, Llorente-Saguer | arXiv [2604.18901](https://arxiv.org/abs/2604.18901) (04/2026) | 12 model 0.5–9B. Chỉ một linear direction, 100 mẫu/lớp, đạt AUROC TB 0.982. **Max-pool trên content tokens và last-token cho ra 2 direction lệch nhau khoảng 73°**, nên nên dùng cả hai làm feature | [V abstract] |
| JBShield, Zhang et al. | **USENIX Security 2025**, [2502.07557](https://arxiv.org/abs/2502.07557) | Theo linear representation hypothesis: jailbreak kích hoạt cả concept "toxic" lẫn concept "jailbreak". Detection acc TB 0.95 trên 5 LLM | [S] |
| Jailbreaking Leaves a Trace, Kadali & Papalexakis | arXiv [2602.11495](https://arxiv.org/abs/2602.11495) (02/2026) | Tensor decomposition trên activations của GPT-J/LLaMA/**Mistral**. Classifier nhẹ, không fine-tune | [V abstract] |
| Constitutional Classifiers++, Cunningham, Wei et al. (Anthropic) | arXiv [2601.04603](https://arxiv.org/abs/2601.04603) (01/2026) | **Cascade 2 tầng**: classifier nhẹ sàng lọc toàn bộ, escalate mẫu đáng ngờ. **Linear probes ensemble với external classifiers**. Chi phí giảm 40×, refusal 0.05% | [V abstract] |
| TaskTracker / "Get my drift?", Abdelnabi et al. | SaTML 2025 ([CISPA](https://cispa.de/en/research/publications/84748-get-my-drift-catching-llm-task-drift-with-activation-deltas)) | Probe trên activation deltas để bắt task drift (indirect PI) | [S] |
| GradSafe (Xie et al.) | ACL 2024, [2402.13494](https://arxiv.org/abs/2402.13494) | Dùng gradient của LLM với tham số safety-critical | [M] |
| Attention Tracker (Hung et al.) | NAACL Findings 2025, [2411.00348](https://arxiv.org/abs/2411.00348) | Theo dõi attention của "important heads" để phát hiện PI, không cần train | [M] |

### 3.2 Detector / guard fine-tune (xem bảng chi tiết trong RESEARCH_V3_GUARDS.md)
- **DataSentinel**, Liu et al., **IEEE S&P 2025**, [2504.11358](https://arxiv.org/abs/2504.11358) [S]: fine-tune LLM detector theo minimax với injected prompt thích nghi. Code ở Open-Prompt-Injection.
- **PromptShield**, Jacob et al., CODASPY 2025, [2501.15145](https://arxiv.org/abs/2501.15145) [S]: benchmark và detector tối ưu ở low-FPR. PromptGuard chỉ bắt được 9.4% attack ở 0.1% FPR.
- **PIGuard/InjecGuard (MOF)**, ACL 2025, [2025.acl-long.1468](https://aclanthology.org/2025.acl-long.1468) [S]: over-defense do trigger words. NotInject có 339 mẫu benign, accuracy các model tụt xuống ~60%.
- **LongPIBench**, Liu, Jia, Gong, Jia, arXiv [2608.28411](https://arxiv.org/abs/2608.28411) (08/2026): benchmark PI cho long context, đánh giá DataSentinel, PromptGuard 1/2, PIShield, DeBERTa/DistilBERT [S: chỉ đọc được metadata, **nội dung chưa kiểm chứng**].
- **Sentinel** (ModernBERT-large), [2506.05446](https://arxiv.org/abs/2506.05446) [S]. Các model card cộng đồng cho thấy **role_play là slice có F1 thấp nhất**: [alvi42](https://huggingface.co/alvi42/prompt-injection-guard-v1) có role_play F1 0.964 [S]. Một ModernBERT guard có FPR 80% trên XSTest "scary-word benign" [S, model card trong kết quả search, chưa mở].
- **Guardrail robustness eval**, [2511.22047](https://arxiv.org/pdf/2511.22047) [S]: 1,445 adversarial prompt. Model tốt nhất đạt 85.3%, kém nhất 38%.

### 3.3 Benchmark, cuộc thi, over-refusal
- **WildJailbreak / WildGuardTest** ([2406.18510](https://arxiv.org/abs/2406.18510), [2406.18495](https://arxiv.org/abs/2406.18495)): adversarial benign và adversarial harmful dùng **cùng tactic wrapper**, chỉ khác intent gốc. Đây chính là cấu trúc pt dài của mình (ERRORS.md: tỉ lệ 50/50 1,824 vs 1,836) [V trong V3].
- **CCFC** ([2508.14128](https://arxiv.org/pdf/2508.14128)) [S]: coi jailbreak là "wrapper bọc một core instruction". Phòng thủ bằng cách tách core rồi đánh giá cả core lẫn full prompt.
  - Hàm ý cho mình [U]: model cần "nhìn thấy" core request. Core thường nằm ở giữa hoặc cuối wrapper, nên context dài và head+tail quan trọng *với Mistral*.
  - Với encoder, max_len 768 đã không giúp, nên encoder không bị giới hạn bởi độ dài.
- **Over-refusal**:
  - OR-Bench (ICML 2025, [2405.20947](https://arxiv.org/abs/2405.20947)) [V qua PMLR listing]: 80k prompt "trông hại nhưng vô hại".
  - XSTest (Röttger et al., NAACL 2024, [M](https://aclanthology.org/2024.naacl-long.301)).
  - Liên quan tới 109 lỗi BENIGN→JAILBREAK ở pt: model phản ứng với "scary words" trong fiction.
- **Cuộc thi PI/jailbreak**:
  - HackAPrompt (EMNLP 2023, [2311.16119](https://arxiv.org/abs/2311.16119)) [M].
  - Tensor Trust (ICLR 2024, [2311.01011](https://arxiv.org/abs/2311.01011)) [M].
  - SaTML 2024 LLM CTF ([2406.07954](https://arxiv.org/abs/2406.07954)) [M].
  - **LLMail-Inject** (Microsoft/ETH/ISTA, SaTML 2025, [2506.09956](https://arxiv.org/abs/2506.09956)) [S]: 208k attack, 839 người tham gia. Attack bị phát hiện có tool-call rate 18.6%, attack không bị phát hiện chỉ 4%, tức có trade-off giữa lẩn tránh và hiệu lực.
  - Nhận xét [U]: các cuộc thi này là *attack-side*, nên ít chuyển giao trực tiếp sang bài classification.
- **Benchmark đánh giá**:
  - JailbreakBench (NeurIPS D&B 2024, [2404.01318](https://arxiv.org/abs/2404.01318)) [M].
  - HarmBench (ICML 2024, [2402.04249](https://arxiv.org/abs/2402.04249)) [M].
  - **JailbreakEval** (NDSS 2025 poster, [2406.09321](https://arxiv.org/abs/2406.09321)) [S]: toolkit có **voting evaluator** kết hợp nhiều evaluator. Đây cũng là một cách hiểu khả dĩ của "decision" (ensemble voting), nhưng với mình voting đã bị reject.
  - Lakera PINT ([github](https://github.com/lakeraai/pint-benchmark)) [M].
  - BIPIA (KDD 2025, [2312.14197](https://arxiv.org/abs/2312.14197)) [M].
- **Đa ngôn ngữ**:
  - MultiJail (ICLR 2024, [2310.06474](https://arxiv.org/abs/2310.06474)) [M], XSafety (ACL Findings 2024, [2310.00905](https://arxiv.org/abs/2310.00905)) [M].
  - PolyGuard ([2504.04377](https://arxiv.org/abs/2504.04377)) [V trong V3].
  - **MLJailDe**, "One Jailbreak, Many Tongues" ([2606.11202](https://arxiv.org/abs/2606.11202), 04/2026) [V abstract]: "relative-distance constraints" kéo các jailbreak cùng intent ở nhiều ngôn ngữ về cùng cluster, kèm imbalance-aware loss. F1 98.5%. **Data tạo bằng back-translation (MT ngoài)**, nên với mình là không hợp lệ. Chỉ ý tưởng loss có thể dùng nếu có cặp song ngữ trong train (twin check trước đó cho thấy không có).
  - Cross-lingual semantic codebooks ([2604.25716](https://arxiv.org/pdf/2604.25716)) [S]: detect bằng embedding thì sụp trên data đa dạng (recall 3–6%).
  - **Không tìm thấy kết quả nào riêng cho tiếng Bồ Đào Nha.**

### 3.4 Label noise và distillation (cho khoảng 40–50 mẫu "mọi model đều sai")
- Zhu et al. 2022, "Is BERT Robust to Label Noise?" [S, [link](https://astro.paperswithcode.com/paper/is-bert-robust-to-label-noise-a-study-on)]: phương pháp chống noise thường **không** giúp BERT. Early stopping là quan trọng nhất vì BERT cuối cùng sẽ ghi nhớ nhãn sai.
- Lukasik et al., ICML 2020 ([PMLR](https://proceedings.mlr.press/v119/lukasik20a.html)) [V listing]: **label smoothing cạnh tranh được với loss-correction** khi có label noise. α lớn hơn noise rate thật vẫn có lợi. Ở noise cao thì lợi thế mất ([2106.04149](https://ar5iv.labs.arxiv.org/html/2106.04149)) [S].
- SaFER (ACL 2023 Industry, [link](https://aclanthology.org/2023.acl-industry.38)) [S]: early stopping không cần nhãn và self-supervised cho BERT với noisy labels.
- Agro & Aldarmaki 2023 [S]: noise thực tế **phụ thuộc vào feature**, không ngẫu nhiên, nên có hại hơn noise ngẫu nhiên. Hàm ý [U]: 40 mẫu nhiễu của mình nhiều khả năng là lỗi dịch hoặc gán nhãn kiểu WildJailbreak. Test cũng có noise tương tự, nên đó là sàn không thể sửa trên test.
- Born-again networks (Furlanello 2018), BAM! (Clark et al. 2019, [1907.04829](https://arxiv.org/abs/1907.04829)) [S]: student cùng kiến trúc, học từ soft labels của teacher, có thể vượt teacher. Teacher annealing giúp. LMSYS 1st place dùng distillation 5-fold [S].
- Confident learning / cleanlab (Northcutt et al., JAIR 2021) [M]: bỏ các mẫu có OOF p(nhãn đã cho) thấp, rồi refit.

### 3.5 Diễn đàn và cộng đồng
- Không tìm thấy thread kỹ thuật cụ thể trên r/LocalLLaMA, r/MachineLearning, X, Zhihu hay CSDN cho bài toán PI/jailbreak classification trong time-box. Search trả về chủ yếu model card trên HF.
- Kaggle Jigsaw Agile Community Rules (10/2025, rule-violation classification): **không tìm được write-up hạng 1** qua search. Có một bài LinkedIn mô tả Qwen embeddings + LR/CatBoost [S]. Pattern "frozen LLM embeddings + classical head" là cùng họ với đề xuất #1.
- Paper so sánh head ([2512.12677](https://arxiv.org/abs/2512.12677)) [V trong V3]: với single-label, last-token head ≥ instruction/label-token. Một nghiên cứu khác ghi nhận LoRA dropout cao có hại, còn rank không có quan hệ rõ với kết quả [S, kết quả search].

---

## 4. Phương pháp chuyển giao được, xếp theo (gain kỳ vọng × xác suất) / chi phí

Tất cả đều **chỉ dùng backbone đã duyệt và `train.csv`**, cộng pseudo-label từ model của mình (đã được chấp nhận).

### #1. Mistral-7B hidden-state linear probe (System Two rẻ) — ưu tiên cao nhất
- **Bằng chứng**: mục 3.1 (PIShield, 2604.18901, JBShield, CC++). Probe vào 7B nắm được "harm semantics" mà encoder 300M–560M thiếu [U].
- **Hợp lệ?**
  - Backbone Mistral-7B-Instruct-v0.3 có trong danh sách. Danh sách ghi "LoRA"; probe trên backbone frozen là trường hợp đặc biệt (r=0, chỉ train head) nên **về tinh thần là tương đương** [U].
  - **Team lead nên ghi rõ trong notebook**: "Mistral-7B-Instruct-v0.3, frozen, linear head trained on train.csv".
  - Nếu muốn an toàn tuyệt đối thì dùng chính LoRA adapter của #2 rồi probe trên đó.
- **Recipe GPU (team lead chạy)**:
  - Đầu vào: prompt bọc text + mô tả nhãn (mục 1.3). Truncation **head 768 + tail 256 token**. 98% pt có ≤ 1700 chars, nên phần lớn vừa trong 1024 [D: chỉ 3.0% pt > 1700 chars].
  - Forward bf16, `output_hidden_states=True`.
  - Lấy 5–6 layer (ví dụ 8, 12, 16, 20, 24, 28 trên 32). Mỗi layer lấy **cả mean-pool trên content tokens lẫn last token** (2604.18901: hai direction khác nhau).
  - Lưu float16, ví dụ 20k × 6 × 2 × 4096 ≈ 2 GB. Hoặc chỉ giữ 3 layer, hoặc PCA 512 trên GPU.
  - Chi phí ước lượng: run Mistral fold-0 (len256, bs16) mất 342 s/epoch cho 11.2k mẫu train. Forward-only cho 20k mẫu ở len1024 khoảng 6–10 phút trên H100, tức ~$0.5–0.8 [U].
- **Phần CPU** (mình/teammate làm được):
  - StandardScaler → LogisticRegression (multinomial, C ∈ {0.01, 0.1, 1}) theo đúng 5 fold `label_language`.
  - Chọn layer và C bằng **nested** (inner fold). Xuất `oof_probs.npy`, `test_probs.npy`.
  - Dùng kết quả theo 2 cách: (a) thêm làm member equal-mean vào blend9; (b) gate theo margin (τ ∈ {0.3, 0.5, 0.8}, α ∈ {0.3, 0.5, 1}), chọn bằng nested CV như `margin_cascade.py`.
- **Gain kỳ vọng**: 0 đến +0.006 OOF [U]. Rủi ro chính: probe 7B frozen có thể chỉ đạt ~0.95–0.96 nếu đứng riêng, nhưng lỗi ít tương quan với encoder.
- **Kill criterion**: nếu acc của probe trên band margin < 0.5 không vượt 0.75 thì bỏ.

### #2. Mistral LoRA v2 (System Two đúng nghĩa)
- **Vì sao v1 yếu [D/U]**: 1 epoch, r16, **len256**. 43% lỗi pt dài hơn 850 chars, tương đương khoảng 256 token [U, ~3.3 chars/token]. Phần core request và đoạn "comece com… aqui está" bị cắt.
- **Recipe**:
  - len 1024 (head 768 + tail 256), 2 epochs, r32/α64, dropout 0.05, target q,k,v,o,gate,up,down.
  - lr 1e-4 (head lr 2e-4), cosine, warmup 5%, bf16, grad ckpt. `pad_token_id` phải được set. `modules_to_save=["score"]`.
  - Prompt có mô tả 3 nhãn (Jev option descriptions).
- **Chi phí**: ước lượng khoảng 4× v1 mỗi epoch, nên 2 epochs khoảng 45 phút/fold H100 ≈ $3 [U]. **Chỉ đủ 1 fold** cho validation, hoặc full-train để predict test.
  - Nếu chỉ có fold0 thì gate/blend chỉ được validate trên fold0 (2,800 dòng, nhiễu khoảng ±0.002).
- **Ghi chú**: thư mục `outputs/runs/mistral_r32_len1024_ep2/` đã có nhưng đang rỗng, có vẻ team lead đã thử. Kiểm tra log trước khi chạy lại.
- **Gain kỳ vọng**: +0.002 đến +0.008 nếu fold0 single đạt ≥ 0.975 [U]. Có thể đây là thứ phân biệt top LB.

### #3. Ensemble self-distillation (born-again) từ blend9
- **Recipe**: student mmBERT-base / wolf (recipe hiện tại), loss = 0.5·CE(hard) + 0.5·KL(soft || p_student), T=1.
  - Soft target cho dòng train lấy từ **blend9 OOF probs**. Cộng thêm test pseudo-label soft (margin ≥ 0.95).
  - Teacher annealing: tỉ lệ soft giảm dần qua các epoch (BAM!).
- **Leak note**: OOF prob của dòng train i đến từ fold model đã thấy fold validation của student. Như vậy student gián tiếp nhận thông tin nhãn của validation, và OOF của student sẽ hơi lạc quan.
  - Giải pháp sạch: với mỗi student fold k, soft target cho train chỉ lấy từ teacher fold models **không chứa fold k**. Muốn vậy cần nested OOF, mà mình không có, nên phải chấp nhận và **trừ đi khoảng 0.001** khi so sánh [U].
- **Gain kỳ vọng**: +0.001 đến +0.003 single model [U]. Ensemble gain nhỏ hơn.

### #4. Label noise hygiene (rẻ, gain nhỏ)
- Label smoothing 0.05–0.1 cho encoder: Lukasik 2020; đây cũng là default trong nhiều lời giải Kaggle [M].
- Refit cuối bỏ khoảng 40 mẫu: những mẫu mọi model đều sai **và** blend conf ≥ 0.95. Không đổi nhãn, chỉ bỏ.
- Không đánh giá được bằng OOF hiện tại, vì phải train lại. **Không đáng một lượt submit riêng**. Chỉ gộp vào run khác.
- **Gain kỳ vọng**: 0 đến +0.001 [U].

### #5. Specialist B-vs-J cho mẫu dài (2 stage kiểu Jev) — ưu tiên thấp
- Train encoder chỉ trên B và J có > 400 chars. Chỉ dùng khi blend đoán B hoặc J với margin < τ.
- Hard-slice up-weighting đã fail, nên kỳ vọng thấp: 0 đến +0.002 [U].

### Không nên làm (lý do)
- **Gọi Jev / TypeSafe API**: không phải backbone được duyệt, và gửi data ra ngoài bị cấm.
- **PolyGuardMix / WildJailbreak / back-translation bằng MT ngoài**: external data, đã bị cấm.
- **Dùng Mistral để dịch test hay trích "core intent"**: đây là output generative của LLM trên test, thuộc vùng xám của luật "no LLM labelling". Chi phí GPU cao, gain chưa rõ.
- Calibration, temperature, per-class threshold: blend đã có ECE 0.006, và per-class multipliers đã fail nested.
- Contrastive cross-lingual (MLJailDe): cần cặp song ngữ, mà twin check cho thấy không có.
- Trộn Mistral v1 hiện có vào blend: +0.0008 fold0, trong nhiễu [D].

---

## 5. Gợi ý cho 3 bài nộp cuối (đầu vào cho team lead) [U]
1. **An toàn**: blend9 equal-mean (OOF 0.9770). Hoặc blend tốt nhất đã có public score.
2. **Có System Two**: blend9 cộng Mistral (probe #1 hoặc LoRA v2 #2), equal-mean hoặc gated. Chỉ nộp nếu nested OOF (hoặc fold0) ≥ +0.002, đồng thời acc trên band margin thấp tốt hơn blend một cách có ý nghĩa (n ≥ 100).
3. **Đa dạng**: blend9 cộng distilled/pseudo round-2 members.
- Public LB chỉ có khoảng 1,800 dòng (SE của macro-F1 khoảng ±0.003). **Không chọn bài nộp theo chênh lệch public < 0.003.**

---

## 6. Claim chưa kiểm chứng (cần kiểm tra trước khi dựa vào)
- "Jev Tysafe" = Jev của TypeSafe AI: suy luận theo tên [U].
- RLCD và kiến trúc của Jev: vendor claim, chưa công bố.
- Accuracy 96.5% so với 89.7% khi có task description: đo bởi bên thứ ba, được trích trong một newsletter.
- Tỉ lệ ~3.3 chars/token của Mistral cho tiếng Bồ Đào Nha [U]. Kiểm tra bằng tokenizer local: `len(tok(text).input_ids)` trên 200 mẫu pt.
- Chi phí GPU của #1 và #2: ngoại suy từ 342 s/epoch của run v1 [U].
- Số liệu PIShield, JBShield, CC++: lấy từ abstract, chưa đọc bảng kết quả.
- Nội dung LongPIBench: chưa đọc, chỉ có metadata.
- Mọi link gắn [M] chưa được mở lại trong phiên này.
- Mức "+0.984 nếu System Two đạt acc 85% ở band < 0.5": ngoại suy tuyến tính, không phải kết quả đo.
