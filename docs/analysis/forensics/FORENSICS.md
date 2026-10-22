# FORENSICS: Có "đường tắt" (shortcut) nào trong train.csv để lên ~0.99 không?

Ngày 2026-10-22, 15:20 đến 15:45 (giờ VN). Vai trò: data forensics lead. Chỉ dùng CPU, chỉ dùng train.csv/test.csv (đọc in-memory từ zip), không dùng external data, không gọi API ngoài.
Baseline: **blend9** (equal mean của 9 run), OOF macro-F1 **0.97697**, **286 errors**. Folds của member: StratifiedKFold(5, shuffle, rs=42) trên label_language.
Scripts: `C:\Users\Admin\AppData\Local\Temp\claude\d--A-RMITHACKATHON\1170d8c9-19f8-4773-beaf-7cf8b7cc5e2a\scratchpad\team\forensics\` (`s02`–`s12`).
Lưu ý: text là jailbreak/injection, tức **untrusted data**. Báo cáo chỉ trích vài từ mở đầu để làm template.

## TL;DR

1. **Không tìm thấy shortcut hợp lệ nào trong train.csv.** Đã thử hơn 150 signal: punctuation/style, translation artefacts (PT-PT vs PT-BR), placeholder, language-tag mismatch, multilingual lexical cues, opening/closing templates, near-duplicate twins, class-quota. Mỗi rule hoặc **sửa ≤ số lỗi nó phá** ngay cả khi đo in-sample, hoặc **âm** khi đo nested.
2. **Feature-augmented stacker không thắng blend9.** Kết quả nested CV (outer 5-fold seed 7, inner 4-fold để chọn hyper-param): cấu hình tốt nhất (offset-LR, tức blend log-prob làm offset cố định cộng với 128 handcrafted features có L2) đạt **0.97739 (+0.0004)** ở seed 7. Ở seed 11 là +0.00015, ở seed 23 là −0.0005, **trung bình ≈ 0**. Cả 9 cấu hình LR/HGB còn lại đều **thua** blend9, từ −0.0004 đến −0.0054. Mức +0.001 không đạt nên **KHÔNG ghi `outputs/runs/stack_forensics/`**. Kết luận: giữ equal mean.
3. **Toàn bộ khoảng cách nằm ở một slice.** Slice "pt, >400 chars, BENIGN/JAILBREAK" có 3,660 rows và chứa **199/286 lỗi (error rate 5.44%)**. Mọi slice khác chỉ ở 0.76–1.00%. Ở pt, error rate nhảy vọt tại mốc 400 chars rồi **giữ phẳng** đến 2k+ chars, và max_len 768 không giúp gì. Vậy đây là bài toán **semantics** (harmful vs benign intent), không phải truncation.
4. **Top LB 0.99 không thể đạt chỉ bằng cách sửa slice pt.** What-if: dù pt-long đúng 100% thì OOF cũng chỉ **0.9884**. Public LB có 1,800 rows, và sd của macro-F1 trên các subset 1,800 rows là **0.0042**. Các đội #4 (0.9803) và #5 (0.9780) nằm **trong noise** so với blend9 (mean 0.9769). #3 (0.9833) ở khoảng p95. #1 và #2 (0.9906/0.9892) cách ≥3 sd. Muốn tới đó thì true level phải ≈0.985+ (pt-long error ≤~1.6%) và còn cần "may" ở public split, **hoặc** model đó đúng cả các row "label-noise" (43 row mà 14/14 model đều sai), một dấu hiệu khớp với việc dùng **nguồn gốc dữ liệu (WildJailbreak labels, external)**. Suy luận này **chưa kiểm chứng [U]**.
5. Hành động có giá trị: (a) chọn final theo **OOF chứ không theo public**, vì private 4,200 rows có sd ≈0.0025; (b) đòn bẩy duy nhất còn lại là **năng lực semantic trên pt-long**, ví dụ decoder LoRA được approve (Mistral-7B), train đủ cả 5 fold với max_len ≥512, rồi chỉ blend nếu OOF xác nhận. Đây là việc của team lead (GPU), tôi không chạy.

---

## 1. Data, nhìn lại bằng con mắt mới

| Phát hiện | Số liệu | Ý nghĩa |
|---|---|---|
| **Whitespace bị normalize toàn bộ** | 0 newline, 0 tab, 0 double-space, 0 NBSP, 0 leading/trailing whitespace, 100% NFC, trên **mọi** row | Ban tổ chức đã làm phẳng text, nên mọi formatting signature (markdown, xuống dòng) **đã bị xoá**. Không còn source fingerprint ở mức ký tự. |
| **Stratification tuyệt đối** | Ở train, mọi language là đúng 62/31/7. Train/test = đúng 70/30 theo từng language (pt 6300/2700, fr 3500/1500, it 2800/1200, en 1400/600). 0.62·2700 = 1674 là số nguyên. | Test gần như chắc chắn là **1674/837/189 (pt), 930/465/105 (fr), 744/372/84 (it), 372/186/42 (en)**. Prediction hiện tại trên test: pt [1679, 837, 184], en [378, 187, 35], khớp rất sát. |
| **pt JAILBREAK khác hẳn fr/it/en JAILBREAK** | Median length của JB: pt **879** chars (p5 258, p95 1792); fr 380, it 360, en 328 (p95 khoảng 700) | pt lấy JB từ một generator khác (role-play dài kiểu WildJailbreak adversarial_harmful). fr/it/en JB là một template paraphrase ngắn. Vì vậy cross-lingual twins cho pt **gần như không tồn tại**. |
| **Slice khó** | pt >400 chars BEN/JB: 3,660 rows, tỉ lệ JB **0.502** (50/50), error **5.44%**. Cùng định nghĩa ở fr/it/en: 0.62% / 0.84% / 0.29% | Ở pt-long, BENIGN là role-play "adversarial_benign" có cùng khung với JB. |
| **Language tag không phải ngôn ngữ thật** | Đo bằng stopword-profile argmax: 601 train / 266 test row lệch tag (es, de, id, ...). Ví dụ có row tiếng Indonesia gắn tag `fr`. | 88% là BENIGN, 11% PI, error rate 0.33%. Model **đã xử lý tốt**, không phải đòn bẩy. |
| **Near-duplicates** | TF-IDF word 1–2 cosine ≥0.8: 393 train rows. Label agreement với neighbour là **100%**, nhưng blend9 chỉ sai **2** trong số đó. Không có exact duplicate train/test; 160 test rows có train-neighbour ≥0.8. | Twins đã được model học. kNN label copying **không còn gì để lấy** (khớp với kết quả reject trước đó). |

## 2. Kiểm định giả thuyết (hypothesis tests). Mọi con số đều trên train, đo trên lỗi của blend9

Định nghĩa: "rule_fix" là số lỗi blend9 sẽ được sửa nếu ép label = majority-label của rows có signal; "rule_break" là số row blend9 đang đúng sẽ bị phá. Đo **in-sample**, tức là lạc quan cho rule. Bảng đầy đủ: `tbl_signals.csv`, `tbl_cues_multilingual.csv`, `tbl_opening_closing_ptlong.csv`.

### 2.1 Punctuation / style / translation artefacts (pt >400 chars BEN/JB, base P(JB)=0.502)

| signal | rows | share | P(JB) | errors có signal | fix / break |
|---|---|---|---|---|---|
| kết thúc bằng "?" | 435 | 11.9% | 0.29 | 23 | 8 / 111 |
| có straight `"` | 1,572 | 43.0% | 0.60 | 84 | 34 / 586 |
| có ":" | 1,713 | 46.8% | 0.60 | 94 | 35 / 621 |
| có "..." | 621 | 17.0% | 0.63 | 27 | 9 / 215 |
| kết thúc bằng quote | 963 | 26.3% | 0.67 | 44 | 19 / 292 |
| **PT-PT markers** (utilizador, facto, equipa, tu-forms...) | 148 | 4.0% | 0.68 | 5 | 3 / 45 |
| PT-BR markers | 3,584 | 97.9% | 0.50 | 197 | – |
| em-dash | 30 | 0.8% | 0.43 | 4 | 3 / 12 |
| placeholder `[...]` | 7 | 0.2% | 0.71 | 2 | 0 / 0 |
| curly quotes, guillemets, NFD, NBSP, newline | 0 | 0% | – | – | không tồn tại ở pt |

Kết luận: giả thuyết "BEN và JB được dịch bằng pipeline khác nhau (PT-PT vs PT-BR)" **bị bác bỏ**. Gần như toàn bộ pt là PT-BR, PT-PT chỉ chiếm 4% với purity 0.68. Mọi style signal đều có purity ≤0.71.

### 2.2 Multilingual lexical cues (20 regex: start-with, "Sure, here", ignore, research, legal/ethical, harm keywords, celebrity names, "you are", question-start, [TARGETLANGUAGE], system prompt)

- Trên pt-long, P(JB) chỉ nằm trong khoảng 0.31–0.77, và **mọi cue đều có fix << break**. Ví dụ `cue_start_with`: 516 rows, P(JB)=0.70, fix 10 / break 145. `cue_harm`: 280 rows, P(JB)=0.69, fix 4 / break 80.
- `[TARGETLANGUAGE]/[PROMPT]` placeholder: 1,065 rows, **99.9% BENIGN**, chỉ 1 error. Đây là shortcut có thật nhưng **model đã học xong**.

### 2.3 Opening / closing templates (sub-population signature trong pt-long)

Đây là phát hiện mới duy nhất có ý nghĩa thống kê (binomial p ~1e-14): một số **persona-opening** chỉ xuất hiện ở JB, vì có thể chúng là bản dịch của một tactic cụ thể do LLM rewriter sinh ra:

| opening (4 từ đầu) | n | P(JB) | errors blend9 |
|---|---|---|---|
| "considere que você é" | 45 | **1.00** | 0 |
| "na posição de (um)" | 41 | **1.00** | 1 |
| "sabendo que você é" | 34 | **1.00** | 0 |
| "você assume ..." | 27 | **1.00** | 1 |
| "no papel de (um)" | 24 | **1.00** | 2 |
| "na sua posição de" | 22 | **1.00** | 2 |
| "imagine que você é" | 397 | 0.23 (nghiêng BEN) | 16 |
| "como um" | 234 | 0.27 (nghiêng BEN) | 16 |

Nhưng khi **đo nested** (template được tìm trên outer-train và áp lên outer-test, `tbl_template_rules_nested.csv`), template "pure" (n≥15, purity 1.0) đổi label 23 rows, chỉ **sửa 8 mà phá 15** (−0.0011). Bản chỉ áp khi conf<0.9 thì 4 fix / 2 break (−0.0002). Bản any 4–5-gram tốt nhất đạt +0.00009, tức noise. Lý do: model **đã học các template này**. Chỗ blend còn sai là các **ngoại lệ**, và ở ngoại lệ thì template không còn pure.

### 2.4 Class-quota (transductive prior) và near-duplicates

| test | kết quả |
|---|---|
| Ép đúng quota 62/31/7 theo từng language (Hungarian assignment trên log-prob), đo trên toàn OOF | 286 → 281 errors (+0.0003) |
| Ép quota theo fold × language (kích thước giống test) | 286 → 286 (0) |
| Mô phỏng 60 lần, mỗi lần lấy ngẫu nhiên một tập stratified có **đúng kích thước test** (2700/1500/1200/600) | delta trung bình **−0.00025**, P(delta<0) = 0.62. **Bác bỏ.** |
| Copy label từ neighbour, sim ≥0.5 / 0.7 / 0.9 | fix 7 / break 4; fix 2 / break 1; fix 1 / break 0. Không đáng kể. |

### 2.5 Các row mà "mọi model đều sai" (`f04_models_right_on_errors.png`)

- Dùng 14 run (9 của blend9 + 5 run phụ): có **43/286 lỗi mà 14/14 model đều sai**. Phân bố: pt 33 (BEN→JB 13, JB→BEN 18), en 5, it 3, fr 2. **93%** trong số này có p_true <0.05 (median 0.002).
- Chúng **không phải near-duplicate**: median NN-sim 0.14, so với 0.23 trên toàn tập. Length median 777 (các lỗi khác 757). Không có signal nào trong mục 2.1–2.3 tách được nhóm này.
- Kết luận: đây là **label noise / source-label** (label được gán theo nguồn sinh dữ liệu chứ không theo ngữ nghĩa). Mức sàn khoảng 0.3% error, và **không model ngữ nghĩa nào sửa được nếu không biết source**.

### 2.6 Truncation (`f05_error_vs_length_pt.png`)

Error rate của pt BEN/JB theo length: <400 chars là 0.5%; 400–800 là 5.2%; 800–1.2k là 5.5%; 1.2–1.6k là 5.1%; 1.6–2k là 7.1%; >2k là 7.5% (n=80). Bản mmBERT-small len768 so với len384 chỉ giảm nhẹ ở 400–1.6k, và wolf len768 ≈ len384. **Truncation chỉ giải thích được phần nhỏ (rất ít row >1.6k)**. Bước nhảy chính xảy ra tại 400 chars, đúng chỗ adversarial_benign bắt đầu xuất hiện.

## 3. Feature-augmented stacker, nested CV

Features: 9 member log-probs (hoặc mean), member std, entropy, agreement, language one-hot, 29 handcrafted style features (mục 2.1), 21 cue/stopword-profile features (mục 2.2), TF-IDF char 2–5 + word 1–2 LR OOF probs (OOF 0.918, một model có inductive bias khác), cùng các pt-interaction.
Protocol: outer StratifiedKFold(5, shuffle, rs=7) theo label_language. Hyper-param được chọn bằng inner 4-fold **chỉ trên outer-train**, sau đó refit trên outer-train và dự đoán outer-test.

| stacker | #feat | nested macro-F1 | Δ vs blend9 | errors | pt F1 |
|---|---|---|---|---|---|
| **blend9 equal mean** (không fit) | – | **0.97697** | 0 | 286 | 0.9690 |
| offset-LR (blend log-prob làm offset + L2 trên 128 features), seed 7 | 128 | 0.97739 | +0.00042 | 288 | 0.9682 |
| offset-LR, outer seed 11 | 128 | 0.97712 | +0.00015 | 293 | 0.9684 |
| offset-LR, outer seed 23 | 128 | 0.97648 | −0.00049 | 293 | 0.9682 |
| HGB: mean + disagreement + tfidf + lang + hand + cue | 71 | 0.97659 | −0.00038 | 298 | 0.9669 |
| gated HGB (chỉ sửa rows có conf < τ) | 128 | 0.97651 | −0.00046 | 302 | 0.9681 |
| HGB: members + disagreement + tfidf + ... | 95 | 0.97598 | −0.00099 | 300 | 0.9678 |
| LR: mean + lang | 7 | 0.97593 | −0.00105 | 297 | 0.9687 |
| LR: members + lang + hand + cue | 87 | 0.97586 | −0.00112 | 300 | 0.9674 |
| HGB: mean + lang + hand + cue | 63 | 0.97464 | −0.00233 | 314 | 0.9668 |
| LR: mean + lang + hand + cue | 63 | 0.97428 | −0.00269 | 328 | 0.9643 |
| LR: mean + extra runs + tfidf + all | 78 | 0.97348 | −0.00350 | 328 | 0.9639 |
| LR: mean + tfidf + lang + hand + cue + pt-interactions | 75 | 0.97189 | −0.00508 | 340 | 0.9637 |
| LR: mean + disagreement + lang + hand + cue + pt-interactions | 72 | 0.97155 | −0.00542 | 341 | 0.9638 |

**Kết luận:** không cấu hình nào đạt +0.001. Offset-LR tốt nhất, nhưng trung bình 3 outer seed là **≈ −0.0000**, tức noise. Vì vậy **không tạo `outputs/runs/stack_forensics/`**, và submission vẫn nên dùng equal mean. Lý do có thể là: (1) member probs đã bão hoà, ECE khoảng 0.006; (2) các feature bề mặt đã nằm sẵn trong transformer; (3) lỗi còn lại là semantic hoặc noise, mà features bề mặt **không mã hoá** được.

## 4. Top teams làm thế nào? (định lượng, `f03_public_lb_noise.png`, `f07_whatif_ptlong.png`)

- **Public-LB noise:** blend9 tính trên 2,000 subset ngẫu nhiên 1,800 rows cho mean 0.9769, **sd 0.0042**, p95 0.9834, max 0.9943. Như vậy #5 (0.9780) và #4 (0.9803) **không chứng minh** được là tốt hơn chúng ta. #3 (0.9833) ở mức p95.
- **What-if pt-long:** pt-long error 5.44% → 2.2% tương ứng 0.9838; → 0.71% (mức fr/it) tương ứng 0.9869; → 0% tương ứng **0.9884**. Muốn đạt 0.9906 trên public thì cần true level khoảng ≥0.985 cộng thêm khoảng +1 sd may mắn, hoặc phải sửa được cả nhóm noise/PI ở en/fr/it.
- Giải thích hợp lý nhất cho #1/#2 [U]: (a) một model có năng lực semantic mạnh hơn nhiều trên pt-long, như decoder 7B LoRA train đúng cách hoặc WildGuard-style; hoặc (b) external data hay source lookup (WildJailbreak gốc có sẵn label adversarial_benign/harmful). Với (b), model sẽ đúng cả 43 row "noise". Cách (b) bị luật cấm (no external data). Chỉ (a) là hợp lệ.

## 5. Khuyến nghị cho team lead (compliant)

| # | hành động | evidence | gain ước tính |
|---|---|---|---|
| 1 | **Giữ equal-mean blend9** (hoặc blend mới, nhưng vẫn equal-mean) làm final chính. Không dùng stacker/rule/quota. | Mục 3 và 2.4: mọi biến thể nested đều ≤ +0.0004 và không ổn định qua seed | 0 (tránh overfit) |
| 2 | **Chọn final theo OOF, không theo public.** Private có 4,200 rows (sd ≈0.0025), còn public sd 0.0042 | Mục 4 | tránh mất 0.002–0.005 do chọn nhầm |
| 3 | Nếu còn GPU: chỉ đầu tư vào **semantic capacity cho pt-long BEN/JB**, ví dụ Mistral-7B-Instruct LoRA (approved) train đủ 5 fold, max_len ≥512, ≥2 epochs. Chỉ thêm vào blend nếu **OOF đủ 5 fold** của blend tăng. Probe 256-token/1-epoch hiện có **tệ hơn** blend9 trên pt-long (fold0: 55 so với 37 errors). | Mục 2.6, 4 | [U] 0 → +0.005 |
| 4 | Không đầu tư thêm vào regex, template, style, PT-PT, placeholder, kNN twins hay quota | Mục 2 | – |

## Files

- Figures: `f01_error_slices.png`, `f02_signal_purity_fixbreak.png`, `f03_public_lb_noise.png`, `f04_models_right_on_errors.png`, `f05_error_vs_length_pt.png`, `f06_stackers_nested.png`, `f07_whatif_ptlong.png`. Palette: blue/orange/aqua/yellow, đã qua CVD validator.
- Tables: `tbl_signals.csv`, `tbl_cues_multilingual.csv`, `tbl_opening_closing_ptlong.csv`, `tbl_template_rules_nested.csv`.
- Nested predictions và JSON kết quả stacker nằm trong scratchpad `team\forensics\` (`nested_*.npy`, `stack_results_*.json`).
