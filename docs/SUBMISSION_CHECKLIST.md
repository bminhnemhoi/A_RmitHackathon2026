# C1 — Checklist tuân thủ & quy trình nộp bài (RMIT Hackathon 2026)

> Vai trò: compliance & submission QA. Cập nhật 22/10/2026 ~15:30 (giờ VN). Hạn chót **18:00**; chốt bài cuối trước **~17:15**.
> Nguồn trích dẫn: `D:\A_RMITHACKATHON\context_rmit.md` (bản dịch tiếng Việt của các trang Kaggle: Overview, Data, Rules, Models)
> và thông tin slide/BTC trong `docs/CONTEXT.md`. Nếu trang Kaggle thay đổi thì **trang Kaggle thắng**.
> Mọi con số dưới đây là đầu ra thật của script audit (`scratchpad/team/compliance/audit_and_blend9.py`, lưu kết quả ở `audit_result.json` cạnh script).

---

## 0. Tóm tắt nhanh

| Ứng viên | File | Hợp lệ (validator) | OOF macro-F1 | pt | Pseudo-label? | Notebook header đúng? |
|---|---|---|---|---|---|---|
| **blend9_equal** (mới) | `outputs/runs/blend9_equal/submission.csv` | VALID (exit 0) | **0.9770** | 0.9690 | có (2/9 run) — **đã khai báo** | **ĐÚNG** — `outputs/runs/blend9_equal/kaggle/RMIT_SGS_Fprinciples_C1.ipynb` |
| blend8_equal | `outputs/runs/blend8_equal/submission.csv` | VALID (exit 0) | 0.9768 | 0.9685 | có (1/8 run) — **KHÔNG khai báo** | THIẾU (xem F2) |
| blend7_equal | `outputs/runs/blend7_equal/submission.csv` | VALID (exit 0) | 0.9764 | 0.9674 | không | ĐÚNG (câu "no pseudo-labels" là đúng với blend7) |

- Khác biệt dự đoán trên test: blend9 và blend7 lệch **16/6000** dòng; blend9 và blend8 lệch **12**; blend8 và blend7 lệch **10**. So với bài đã nộp (blend_mms_xlmrl, public 0.9635): blend9 lệch 64 dòng, blend7 lệch 58 dòng.
- Paired bootstrap trên OOF (1000 lần): blend9 − blend7 = +0.0006 (khoảng 90%: −0.0005…+0.0018, P(>0)=0.84); blend9 − blend8 = +0.0001 (P(>0)=0.57).
  → blend8 không hơn blend9 một cách có ý nghĩa, **không cần tốn slot cho blend8**.
- Còn **4/6** lượt nộp (đã dùng 2: một blend 2 model public 0.9635, một upload bị lỗi; hãy kiểm tra lại bộ đếm trên tab Submissions).

---

## 1. Checklist yêu cầu của BTC (kèm trích dẫn)

Ký hiệu: **[BB]** = bắt buộc, **[KN]** = khuyến nghị mạnh của BTC. Trạng thái tính trên bài blend9 + notebook mới.

| # | Yêu cầu | Trích dẫn nguyên văn (context_rmit.md) | Trạng thái |
|---|---|---|---|
| R1 [BB] | Chỉ dùng backbone trong danh sách duyệt | "Bạn phải xây dựng giải pháp của mình trên một trong những mô hình được chấp thuận sau đây. […] — không được phép sử dụng bất kỳ mô hình xương sống được huấn luyện trước nào khác." | ĐẠT: mmBERT-small, mmBERT-base, XLM-R large, wolf-defender (4/10 model được duyệt). Mistral LoRA **không** nằm trong blend. |
| R2 [BB] | Fine-tune trên `train.csv` | "Hãy tinh chỉnh nó trên train.csv." / "Huấn luyện trên train.csv(14.000 hàng, cột id, text, language, label) và dự đoán trên test.csv(6.000 hàng)." | ĐẠT: mọi run fine-tune đầy đủ trên train.csv, 5-fold StratifiedKFold(label×language, seed 42). |
| R3 [BB] | Ghi rõ model ở **đầu** notebook | "Hãy ghi rõ mẫu bạn đã chọn ở đầu cuốn sổ tay." | ĐẠT với notebook blend9 mới (ô đầu tiên liệt kê 4 backbone + Kaggle slug). Notebook cũ `RMIT_SGS_Fprinciples_C1.ipynb` **KHÔNG ĐẠT** (F3). |
| R4 [BB] | Tên notebook theo quy ước | "Quy ước đặt tên Notebook/Kernel : Phải là <university>_<campus>_<team>_C<challenge> nơi <challenge>∈ {1} và <campus> ∈ {SGS, HNI, MEL, MYS}." và "Chỉ sử dụng chữ cái cho <university> và <campus>; chữ cái/số/ _/ - cho <team>." | Tiêu đề trên Kaggle phải **đúng chính xác `RMIT_SGS_Fprinciples_C1`**. Không dùng tiêu đề kiểu `RMIT_SGS_Fprinciples_C1_blend7_equal` (sai mẫu vì không kết thúc bằng `_C1`) (F4). |
| R5 [BB] | Lập đội trên Kaggle | "Hãy lập nhóm của bạn trên Kaggle hoặc tham gia một nhóm hiện có." + Quy tắc 5b: "bạn chỉ được tham gia hoặc thành lập một Nhóm." "Bạn phải xác nhận tư cách thành viên Nhóm của mình […] bằng cách trả lời tin nhắn thông báo Nhóm" | Kiểm tra tab Team: tên đội `RMIT_SGS_Fprinciples_C1` (theo slide), mọi thành viên đã accept. |
| R6 [BB] | Định dạng CSV `id,label` | "Tệp phải chứa chính xác 6.000 hàng cộng với tiêu đề , một tiêu đề cho mỗi id mục trong test.csv." / "label phải chính xác là một trong các giá trị BENIGN, JAILBREAK, PROMPT_INJECTION." / "Hãy gửi một tệp CSV có cấu trúc tương tự như sau sample_submission.csv: 6.000 hàng, mỗi hàng một bài kiểm tra id." | ĐẠT cả 3 file: header `id,label`, 6000 dòng, 2 cột, id duy nhất, **đúng thứ tự sample**, chỉ 3 nhãn hợp lệ, 0 ô rỗng, không BOM (xuống dòng CRLF — Kaggle chấp nhận). |
| R7 [BB] | Tối đa 6 lượt nộp | "Quan trọng: Bạn chỉ có 6 lần nộp bài , vì vậy hãy lên kế hoạch thí nghiệm cẩn thận" | Còn 4. Mỗi lượt phải có lý do (mục 4). |
| R8 [BB] | Không gán nhãn thủ công / dự đoán của con người trên test | Quy tắc 4b: "Bài dự thi không được sử dụng hoặc kết hợp thông tin từ việc dán nhãn thủ công hoặc dự đoán của con người về tập dữ liệu xác thực hoặc bản ghi dữ liệu thử nghiệm." | ĐẠT: không có nhãn người/LLM. Pseudo-label là dự đoán máy của chính các model được duyệt (trưởng nhóm đã chấp nhận) → **phải khai báo công khai** trong notebook (đã làm ở blend9). |
| R9 [BB] | Không dùng dữ liệu ngoài | BTC nói rõ (slide/trực tiếp, xem CONTEXT). | ĐẠT: grep `src/ modal_jobs/ notebooks/ scripts/` không có `load_dataset`, URL, API ngoài; nhãn huấn luyện chỉ từ train.csv. Trọng số backbone tải từ repo HF gốc cùng ID với bản `aisuko/*` trên Kaggle (F5). |
| R10 [BB] | Một tài khoản Kaggle / người | Quy tắc 5a: "Bạn chỉ được phép nộp bài dự thi bằng một tài khoản Kaggle.com duy nhất. Bạn sẽ bị loại nếu nộp bài dự thi thông qua nhiều hơn một tài khoản Kaggle" | **Lưu ý**: "tài khoản mới khi hết credit" chỉ áp dụng cho **Modal (GPU)**. Tuyệt đối **không** tạo/dùng tài khoản Kaggle thứ hai để nộp thêm. |
| R11 [BB] | Không chia sẻ code/dữ liệu riêng ra ngoài đội | Quy tắc 5d: "Không được chia sẻ riêng tư bên ngoài Nhóm." | Dataset Kaggle chứa xác suất để ở chế độ **Private**, chỉ chia sẻ cho thành viên đội. Repo GitHub public: chưa đẩy code model trước 18:00 (AGENTS.md). |
| R12 [BB] | Chỉ dùng mã nguồn mở giấy phép OSI | Quy tắc 6c: "bạn chỉ được sử dụng mã nguồn mở được cấp phép theo giấy phép được Sáng kiến Mã nguồn Mở (Open Source Initiative) phê duyệt" | ĐẠT: torch/numpy/pandas/scikit-learn (BSD), transformers/huggingface_hub/sentencepiece (Apache-2.0). |
| R13 [BB] | Nộp trước hạn | Quy tắc 4a: "Bài dự thi phải được nhận trước bất kỳ hạn chót nộp bài nào được nêu trên Trang web của Cuộc thi." | Hạn 18:00 VN; chốt lựa chọn trước 17:15 để có đệm. |
| R14 [BB] | Chọn bài cuối | Quy tắc 18c: "\"Bài nộp Cuối cùng\" là Bài nộp được người dùng chọn, hoặc được Kaggle tự động chọn trong trường hợp người dùng không chọn" + slide: tối đa 3. | Phải tick chọn thủ công (mục 3, bước 6). |
| R15 [BB] | Xếp hạng theo private LB | Quy tắc 7a: "Người thắng cuộc tiềm năng được xác định hoàn toàn dựa trên thứ hạng trên Bảng xếp hạng Riêng tư" + Overview: "Hãy tập trung vào khả năng khái quát hóa mạnh mẽ , chứ không phải là việc quá khớp dữ liệu trên bảng xếp hạng." | Chọn theo OOF, không theo public LB (public chỉ ~1800 dòng). |
| R16 [KN] | Nộp qua notebook | Trang Rules: "Nộp vở ghi chép để được điểm cao hơn." | Khuyến nghị nộp từ Output của notebook `RMIT_SGS_Fprinciples_C1` (mục 3, bước 4A). |
| R17 [BB] | Giấy phép dữ liệu | Trang Data: "Giấy phép CC BY-NC-SA 4.0" | Không công khai train/test ở bất kỳ đâu. Bundle blend9 **không** chứa text, chỉ có xác suất + code. |
| — | Điểm tối đa C1 khi có bài hợp lệ đầu tiên | "Bạn sẽ nhận được điểm tối đa của Thử thách 1 ngay khi gửi thành công kết quả hợp lệ đầu tiên" | ĐÃ ĐẠT (public 0.9635 đã có điểm). |

---

## 2. Kết quả audit chi tiết & các điểm mâu thuẫn

### 2.1 Kiểm tra dữ liệu & thứ tự (đều ĐẠT)
- `test.csv` id **trùng hoàn toàn thứ tự** với `sample_submission.csv` (quan trọng vì `test_probs.npy` lưu theo thứ tự test.csv, còn submission ghi theo thứ tự sample).
- `outputs/train_labels.npy` và `train_langs.npy` khớp thứ tự train.csv. `_sample.csv` của cả 9 run khớp sample trong zip.
- Test và train có **0** câu trùng chính xác và **0** câu trùng sau chuẩn hoá (NFKC + lowercase + gộp khoảng trắng) → không có rò rỉ qua bản sao; độ lạc quan OOF của run pseudo chỉ còn ở mức "gần nghĩa" (nhẹ).
- Số dòng pseudo tái lập đúng: blend6_equal có 5121 dòng conf ≥ 0.98 (mmb_pseudo_ep4), blend8_equal có 5142 dòng (wolf_pseudo_ep4).
- blend7/blend8: `test_probs.npy`, `oof_probs.npy` và `submission.csv` tái tạo **đúng 100%** từ trung bình đều các run thành viên.
- Chạy thử (dry-run) 3 notebook trên cây `/kaggle/input` giả lập: output của cả blend7, blend8, blend9 **trùng từng byte nhãn** với file submission tương ứng.

### 2.2 Các điểm cần sửa / rủi ro

| ID | Mức | Vấn đề | Việc cần làm |
|---|---|---|---|
| F1 | **CAO** | `scripts/make_kaggle_bundle.py` ghi cứng câu "**no pseudo-labels**" vào header. Nếu ai dùng nó để sinh notebook cho blend8/blend9 thì notebook sẽ **khai sai** (blend8/9 có run pseudo-label). | **Không** dùng script đó cho blend8/9. Dùng notebook mới `outputs/runs/blend9_equal/kaggle/RMIT_SGS_Fprinciples_C1.ipynb` (có khai báo pseudo-label rõ ràng). |
| F2 | **CAO** | `notebooks/RMIT_SGS_Fprinciples_C1_blend8_equal.ipynb`: có `mmb_pseudo_ep4` nhưng **không khai báo** pseudo-label; không có dòng "Data"; trỏ tới code "in the team repository" (repo public chưa có code) và nhắc `train_lora.py` (Mistral) dù không có run nào dùng; bundle `outputs/kaggle_bundle/blend8_equal` **thiếu** `train_modal.py`. | Không nộp blend8 (blend9 trội hơn hoặc ngang). Nếu vẫn nộp thì phải sửa header như blend9. |
| F3 | **CAO** | `notebooks/RMIT_SGS_Fprinciples_C1.ipynb`: tiêu đề còn placeholder `RMIT_<CAMPUS>_Fprinciples_C1`; header ghi "Model used: `jhu-clsp/mmBERT-small`" (notebook baseline 1 split, có class multipliers) — **mâu thuẫn** với mọi bài ensemble đang nộp. Nếu notebook Kaggle tên `RMIT_SGS_Fprinciples_C1` đang ở phiên bản này thì phần "model ở đầu notebook" sai. | Tạo **phiên bản mới** của notebook Kaggle `RMIT_SGS_Fprinciples_C1` bằng nội dung notebook blend9 (hoặc blend7). |
| F4 | TRUNG BÌNH | Tên file cục bộ có hậu tố (`…_C1_blend7_equal.ipynb`). Nếu tiêu đề Kaggle cũng mang hậu tố thì sai quy ước `<university>_<campus>_<team>_C1`. | Trên Kaggle luôn đặt tiêu đề đúng `RMIT_SGS_Fprinciples_C1`; phân biệt các blend bằng **Version**, không bằng tên. |
| F5 | THẤP | Trọng số backbone tải bằng `snapshot_download` từ repo HF gốc (cùng model ID với bản `aisuko/*`), không phải từ Kaggle Models. | Đã ghi cả HF ID và Kaggle slug ở đầu notebook blend9. Nếu BTC hỏi: cùng model, cùng trọng số công khai. |
| F6 | THẤP | OOF của 2 run pseudo hơi lạc quan (teacher đã thấy mọi fold). | Đã ghi rõ trong header + `summary.json`. blend7 (không pseudo) là phương án dự phòng sạch 100%. |
| F7 | THẤP | Notebook blend7/blend8 cũ không assert thứ tự test == sample (đúng với dữ liệu này nên kết quả vẫn đúng). | Notebook blend9 mới đã có assert này. |

---

## 3. Quy trình nộp từng bước (chỉ các bước BTC yêu cầu)

Bài mặc định: **blend9_equal**. Dự phòng sạch (không pseudo): **blend7_equal**.

**Bước 1 [BB] — Kiểm tra file trên máy (đã chạy, kết quả thật):**
```bash
python -I scripts/validate_submission.py --sample outputs/runs/blend9_equal/_sample.csv --submission outputs/runs/blend9_equal/submission.csv
# VALID submission: 6000 rows
#   BENIGN             3733  (62.2%)
#   JAILBREAK          1862  (31.0%)
#   PROMPT_INJECTION    405  (6.8%)
```
(blend7: 3739 / 1860 / 401 — VALID.)

**Bước 2 [BB] — Đội (R5, R10):** Kaggle → competition `rmit-hackathon-2026` → tab **Team**: tên đội đúng `RMIT_SGS_Fprinciples_C1`, đủ thành viên đã accept. Chỉ nộp bằng tài khoản Kaggle của chính thành viên trong đội.

**Bước 3 [BB] — Notebook đúng tên + model ở ô đầu (R3, R4):**
1. Kaggle → **Datasets → New Dataset** → tải lên **toàn bộ** file trong `outputs/runs/blend9_equal/kaggle/dataset/` (17 file: 9 `*__test_probs.npy`, 2 teacher probs trong thư mục con `pseudo_teachers/`, `manifest.json`, `train_modal.py`, `metrics.py`, `ensemble.py`, `validate_submission.py`, `dataset-metadata.json`) → tên `fprinciples-c1-blend9-equal` → **Private** (R11). Không có text train/test trong bundle. Nếu giao diện web làm phẳng thư mục con thì vẫn chạy được (notebook không đọc teacher probs, chúng chỉ để kiểm tra lại).
2. Mở notebook Kaggle `RMIT_SGS_Fprinciples_C1` (nếu đã có thì sửa nó để tạo version mới; nếu chưa có: competition → **Code → New Notebook**) → **File → Import Notebook** → chọn `outputs/runs/blend9_equal/kaggle/RMIT_SGS_Fprinciples_C1.ipynb`.
3. Tiêu đề notebook: đúng **`RMIT_SGS_Fprinciples_C1`**. Input: competition data `rmit-hackathon-2026` + dataset `fprinciples-c1-blend9-equal`. Accelerator: None (CPU là đủ).
4. Ô đầu tiên phải còn nguyên: danh sách 4 backbone được duyệt + khai báo pseudo-label.
5. **Save Version → Save & Run All**. Khi chạy xong, log phải in `BENIGN 3733 / JAILBREAK 1862 / PROMPT_INJECTION 405` và Output có `submission.csv`.

**Bước 4 [BB] — Nộp CSV (R6, R7).** Chọn **một** trong hai cách:
- **4A [KN — R16 "Nộp vở ghi chép để được điểm cao hơn"]:** competition → **Submit Prediction** → tab **Notebook** → chọn `RMIT_SGS_Fprinciples_C1`, đúng Version vừa chạy, file `submission.csv` → Submit. (Hoặc từ trang notebook → Output → Submit.)
- **4B [tối thiểu]:** competition → **Submit Prediction** → upload `outputs/runs/blend9_equal/submission.csv`. Mô tả: `blend9_equal: 9-run equal mean (mmBERT-s/b, XLM-R-L, wolf), OOF 0.9770, 2 runs pseudo-labelled`.

**Bước 5 [BB] — Xác nhận:** tab **Submissions** hiện điểm (không có "Error"). Nếu lỗi: tải lại file, chạy lại Bước 1, đừng sửa tay CSV.

**Bước 6 [BB] — Chọn bài cuối (R14, R15) trước 17:15:** tab **Submissions** → tick **Select** (tối đa 3 theo slide; Kaggle hiển thị giới hạn thật). Gợi ý của QA (chọn theo OOF, không theo public):
1. **blend9_equal** — OOF cao nhất 0.9770.
2. **blend7_equal** — OOF 0.9764, **không dùng pseudo-label** → phòng trường hợp BTC không chấp nhận pseudo-label trên text test; chỉ lệch blend9 16 dòng.
3. Slot thứ 3: bài tốt nhất mà team decision chốt (nếu có run mới đã kiểm định nested), nếu không thì bài đã nộp có điểm public cao nhất.

Nếu nộp blend7 qua notebook: dùng `notebooks/RMIT_SGS_Fprinciples_C1_blend7_equal.ipynb` + dataset từ `outputs/kaggle_bundle/blend7_equal/` (có `train_modal.py`), nhưng **đặt tiêu đề Kaggle `RMIT_SGS_Fprinciples_C1`** (một version khác của cùng notebook). Header blend7 hiện tại đúng sự thật.

---

## 4. Bước TÙY CHỌN (BTC không yêu cầu — làm nếu còn thời gian)

- Ghi mỗi lượt nộp vào `docs/SUBMISSIONS.md` (slot, giờ, version notebook, OOF, public LB) và `experiments/log.csv`.
- Đính kèm thêm notebook huấn luyện (ví dụ `notebooks/train_kaggle.py` chuyển sang ipynb) cho giám khảo xem; **phải sửa tiêu đề/ô đầu cho đúng model** trước khi đính kèm.
- Kiểm tra checksum trọng số HF và bản `aisuko/*` trên Kaggle (F5).
- Sau 18:00 mới đẩy code model lên GitHub public (AGENTS.md / quy tắc 6).

## 5. Những việc KHÔNG được làm

- Không tạo tài khoản Kaggle thứ hai để có thêm lượt nộp (R10 → bị loại). Tài khoản GPU mới (Modal) là chuyện khác.
- Không gán nhãn tay hay nhờ LLM đọc/gán nhãn test (R8); không gửi text của cuộc thi lên API ngoài.
- Không dùng dữ liệu ngoài, không dùng backbone ngoài danh sách (R1, R9).
- Không sửa tay CSV; không chọn bài cuối chỉ dựa vào public LB (R15).
- Không dùng `scripts/make_kaggle_bundle.py` cho blend có pseudo-label (F1).

## 6. File liên quan

- `outputs/runs/blend9_equal/{submission.csv, summary.json, oof_probs.npy, test_probs.npy, _sample.csv}`
- `outputs/runs/blend9_equal/kaggle/RMIT_SGS_Fprinciples_C1.ipynb` và `outputs/runs/blend9_equal/kaggle/dataset/`
- Script QA: `C:\Users\Admin\AppData\Local\Temp\claude\d--A-RMITHACKATHON\1170d8c9-19f8-4773-beaf-7cf8b7cc5e2a\scratchpad\team\compliance\` (`audit_and_blend9.py`, `make_blend9_kaggle.py`, `dryrun_notebook.py`, `audit_result.json`)
