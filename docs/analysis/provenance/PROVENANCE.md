# PROVENANCE: Nguồn gốc dữ liệu (data provenance) của rmit-hackathon-2026

Ngày: 2026-10-22, khoảng 14:10 đến 14:40 (giờ Việt Nam). Giới hạn thời gian 20 phút nên phân tích chưa đầy đủ; xem mục Hạn chế (limitations).
Scripts: `C:\Users\Admin\AppData\Local\Temp\claude\d--A-RMITHACKATHON\1170d8c9-19f8-4773-beaf-7cf8b7cc5e2a\scratchpad\prov_scripts\` (dl.py, dl2.py, load.py, match.py, report.py, clean.py, cues.py).
Dữ liệu public đã tải: `...\scratchpad\ext_data\`. Dữ liệu cuộc thi KHÔNG được upload đi đâu cả.

> **CẢNH BÁO RÒ RỈ NHÃN (label leakage):** KHÔNG được dùng các nguồn public để tra nhãn cho **TEST**, ví dụ "nếu text test khớp yanismiraoui thì gán PROMPT_INJECTION". Làm vậy là label leakage / gian lận (cheating). Có 273/6000 item test (4.6%) khớp chính xác với một nguồn public, nên rủi ro này có thật.
> Còn việc dùng các nguồn public làm **dữ liệu TRAIN bổ sung (extra training data)** là một **câu hỏi về luật thi (rules question) riêng**, phải hỏi ban tổ chức (organisers) trước. Tài liệu này không cho phép làm điều đó.

## 1. Trạng thái truy cập (gating) của các nguồn

| Nguồn (HF dataset) | Gated? | Tải về? | Ghi chú |
|---|---|---|---|
| deepset/prompt-injections | không | có | 662 dòng, label 0/1, có tiếng Đức (German) |
| JasperLS/prompt-injections | không | có | bản sao (duplicate) của deepset, bị loại khỏi thống kê |
| jackhhao/jailbreak-classification | không | có | benign/jailbreak |
| rubend18/ChatGPT-Jailbreak-Prompts | không | có | 79 prompt kiểu DAN |
| Lakera/gandalf_ignore_instructions | không | có | 1000 dòng, không có nhãn |
| TrustAIRLab/in-the-wild-jailbreak-prompts | không | có | các config jailbreak_* và regular_* |
| yanismiraoui/prompt_injections | không | có | 1034 prompt injection **đa ngôn ngữ (multilingual)** |
| xTRam1/safe-guard-prompt-injection | không | có | label 0/1 |
| reshabhs/SPML_Chatbot_Prompt_Injection | không | có | System Prompt + User Prompt, label 0/1 |
| jayavibhav/prompt-injection | không | có | 327k dòng, nguồn tổng hợp (aggregator); chỉ dùng để so khớp chính xác (exact match) |
| GuardrailsAI/detect-jailbreak | không | có | tổng hợp từ verazuo (= TrustAIRLab) và cais/wmdp |
| **allenai/wildjailbreak** | **gated (auto)** | **không** | không truy cập, không tìm bản mirror (để tránh lách gating) |
| hackaprompt/hackaprompt-dataset | gated | không | |
| qualifire/prompt-injections-benchmark, Mindgard/evaded-..., markush1/LLM-Jailbreak-Classifier | gated | không | |
| geekyrakshit/prompt-injection-dataset, imoxto/..., Lakera/mosscap, lmsys/toxic-chat | không | không (thiếu thời gian) | ứng viên cho lần kiểm tra sau |
| verazuo/jailbreak_llms, deepset/prompt-injections-german, Harelix/... | không tồn tại / 401 | không | |

## 2. Kết quả so khớp (matching)

Cách làm:
- **Chuẩn hoá (normalisation):** NFKC, chuyển chữ thường (lowercase), thay các ký tự không phải chữ/số (non-word) bằng khoảng trắng, gộp khoảng trắng.
- **So khớp chính xác (exact match)** trên **toàn bộ** train+test, mọi ngôn ngữ.
- **So khớp mờ (fuzzy match)** chỉ trên tập EN: TF-IDF char_wb 3-5, cosine. Có 47k text nguồn duy nhất; jayavibhav chỉ giữ các dòng khớp chính xác.

Kết quả tổng quát:
- Exact match: 889 item. Fuzzy ≥0.8 chỉ thêm 3 item, ≥0.6 thêm 32 item. Như vậy ban tổ chức **copy nguyên văn (verbatim)** chứ không paraphrase các text EN lấy từ nguồn public.
- Theo split: train 619/14000 (4.4%), test 273/6000 (4.6%).
- Theo ngôn ngữ: EN 26 đến 29%, fr 3%, it 2 đến 3%, pt 1%.
- Theo nhãn train: BENIGN 284/8680 (3.3%), **JAILBREAK 0/4340 (0%)**, PROMPT_INJECTION 335/980 (34%). Riêng EN PI là 87/98 (89%).

### Nhãn của nguồn và nhãn cuộc thi (chỉ tính train, exact match)

| Nguồn / nhãn nguồn | BENIGN | PROMPT_INJECTION | JAILBREAK | test khớp | Ánh xạ (mapping) |
|---|---|---|---|---|---|
| yanismiraoui (tất cả là injection) | 0 | **244** | 0 | 97 | **xác định (deterministic)** → PI; fr 93, it 90, pt 59, en 2 |
| reshabhs SPML, injection=1 | 0 | **74** | 0 | 32 | xác định → PI |
| deepset label=1 | 0 | 8 | 0 | 2 | xác định → PI (gồm cả mảnh tiếng Đức nằm trong fr/it/pt) |
| xTRam1 label=1 / label=0 | 2 / 7 | 8 / 0 | 0 | 16 / 1 | gần xác định (nguồn tổng hợp có nhiễu) |
| jayavibhav label=1 / label=0 | 2 / 7 | 16 / 0 | 0 | 18 / 1 | gần xác định |
| TrustAIRLab **regular** (jailbreak=False) | **276** | 0 | 0 | 125 | xác định → BENIGN |
| TrustAIRLab **jailbreak** (=True), jackhhao `jailbreak`, rubend18 | **6** | 0 | **0** | 11 | **→ BENIGN!** Các prompt kiểu DAN "in-the-wild" KHÔNG thuộc lớp JAILBREAK |

Diễn giải:
- Nhãn cuộc thi là **nhãn theo nguồn (source-level label)**, không phải nhãn được gán lại theo ngữ nghĩa (re-annotated semantics).
  - Các nguồn PI public → PROMPT_INJECTION.
  - Nguồn "regular" (prompt người dùng thật) → BENIGN.
  - Lớp JAILBREAK không lấy từ bất kỳ nguồn jailbreak public ungated nào. Không có item JAILBREAK nào khớp nguồn public, kể cả khi so mờ với ngưỡng 0.6.
- Một số prompt jailbreak "in-the-wild" lại mang nhãn BENIGN. Đây là **nhiễu nhãn (label noise)** hoặc là định nghĩa riêng của ban tổ chức (JAILBREAK = adversarial_harmful kiểu WildJailbreak). Với 6 item thì không ảnh hưởng nhiều.

## 3. Item không phải tiếng Anh: dấu hiệu bất biến theo ngôn ngữ (language-invariant cues)

- Các item PI fr/it/pt khớp **chính xác** với yanismiraoui, vốn đã đa ngôn ngữ sẵn. Vậy các item này **không do ban tổ chức dịch máy**, mà được lấy nguyên từ nguồn.
  - yanismiraoui giải thích được 38% PI fr (93/245), 46% PI it (90/196) và 13% PI pt (59/441).
- Các mảnh tiếng Đức (German fragments) khớp chính xác deepset hoặc jayavibhav (fr 5, it 1, pt 1). Nghĩa là **nhãn ngôn ngữ (language tag) không phải lúc nào cũng đúng ngôn ngữ thật**: text tiếng Đức gốc vẫn được gắn tag fr/it/pt.
- Cách làm với dấu hiệu bất biến: số ≥2 chữ số, từ VIẾT HOA (DAN...), CamelCase (BuyBot, TechHelper), URL, `{placeholder}`, `func()`. Chỉ dùng các token hiếm (df<200), yêu cầu ≥3 token chung và độ phủ IDF ≥0.7.
  - Chỉ 638/18000 item non-EN có ≥3 token hiếm như vậy; 164 item link được tới nguồn public, chủ yếu là GuardrailsAI/TrustAIRLab regular với nhãn BENIGN (95 BENIGN, 11 JAILBREAK, 13 BENIGN từ is_jailbreak=True).
  - Độ tin cậy thấp, chỉ mang tính gợi ý. Cần đọc tay (manual review) mới kết luận được.
- Liên kết chéo ngôn ngữ ngay trong cuộc thi (non-EN → EN, cùng phương pháp): 35 cặp; trong 21 cặp cả hai đều có nhãn thì nhãn trùng 95%. Có 11 cặp nối test với train.
  - Hệ quả: các bản dịch của cùng một item gốc có thể nằm ở cả train lẫn test, nên CV ngẫu nhiên sẽ cho điểm lạc quan quá mức (optimistic CV).

## 4. Kết luận: cách dataset có thể đã được xây dựng (inferred construction recipe)

1. **Nguồn (sources) → nhãn:**
   - PROMPT_INJECTION = deepset + SPML + xTRam1 / jayavibhav (EN) + yanismiraoui (đa ngôn ngữ). Phần còn lại (66% PI train) đến từ nguồn chưa xác định; có thể là các nguồn chưa tải (geekyrakshit, imoxto, mosscap/gandalf) hoặc bản dịch máy (MT) của các nguồn trên.
   - JAILBREAK = gần như chắc chắn là **WildJailbreak `adversarial_harmful`**. Không xác minh trực tiếp được vì allenai/wildjailbreak là gated. Bằng chứng gián tiếp: văn phong "revised adversarial prompt", "As a mischievous prompt writing language model", độ dài 300 đến 900 ký tự với max khoảng 950 (fr/it/en), và 0% khớp với mọi nguồn jailbreak public.
   - BENIGN = TrustAIRLab/verazuo **regular** (~3% train; ~20% BENIGN EN) + WildJailbreak `adversarial_benign` (role-play vô hại) + có thể thêm vanilla_benign hoặc câu hỏi ngắn kiểu toxic-chat (chưa kiểm tra).
2. **Tỉ lệ cố định (stratified):** 62/31/7 cho mọi ngôn ngữ, nên dữ liệu được lấy mẫu theo quota cho từng ngôn ngữ và từng nhãn.
3. **Dịch (translation):** item EN được dịch máy sang fr/it/pt, trừ các nguồn vốn đã đa ngôn ngữ (yanismiraoui, mảnh tiếng Đức của deepset) được giữ nguyên.
4. **Ước tính tỉ trọng WildJailbreak:** khoảng 31% (toàn bộ JAILBREAK) cộng với phần lớn BENIGN role-play. Tổng có thể vào khoảng **55 đến 80% dataset**. Đây là suy luận, chưa kiểm chứng (unverified).
5. **Phần giải thích được bằng nguồn ungated:** khoảng 4.5% số item (train 4.4%, test 4.6%), tập trung ở EN và PI.

## 5. Hệ quả cho mô hình (modelling implications)

- **Phân biệt JAILBREAK với BENIGN là bài toán phân biệt adversarial_harmful và adversarial_benign của WildJailbreak.** Hai nhóm có văn phong role-play giống nhau, nên mô hình phải dựa vào **ý định có hại (harmful intent)** trong nội dung chứ không dựa vào hình thức jailbreak. Các prompt kiểu DAN "in-the-wild" lại mang nhãn BENIGN.
- PI có đặc trưng nguồn rõ: "ignore previous instructions", "System Prompt + User Prompt" kiểu SPML, câu hỏi kiểu "hãy tiết lộ system prompt" (yanismiraoui). Mô hình dễ học được lớp này, nhưng các mẫu đa ngôn ngữ hoặc tiếng Đức có thể mang tag ngôn ngữ sai, nên **không nên phụ thuộc vào cột `language`**.
- Nên dùng **CV theo nhóm (group-aware CV)**: gom các bản dịch / cụm gần trùng (near-duplicate clusters) theo dấu hiệu bất biến hoặc embedding đa ngôn ngữ (multilingual embedding) để tránh rò rỉ giữa các fold.
- **Không** tra nhãn test qua nguồn public (label leakage). Dùng nguồn public làm dữ liệu train bổ sung chỉ khi **ban tổ chức cho phép rõ ràng**. Nếu được phép, nguồn có giá trị nhất là WildJailbreak (gated, phải tự chấp nhận điều khoản trên HF).

## 6. Hạn chế (limitations)

- Fuzzy matching chỉ chạy cho EN. Non-EN chỉ có exact match và dấu hiệu bất biến; chưa có so khớp xuyên ngôn ngữ (cross-lingual) bằng embedding hoặc dịch ngược (back-translation).
- Chưa tải geekyrakshit, imoxto, mosscap, toxic-chat; các nguồn này có thể giải thích thêm phần PI và BENIGN còn thiếu.
- Không xác minh trực tiếp được WildJailbreak vì nguồn này gated.
