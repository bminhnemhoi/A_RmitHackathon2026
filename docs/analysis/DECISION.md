# Decision layers trên blend9: đánh giá nested CV

Ngày 2026-10-22. Vai trò: decision-mechanism specialist (chỉ dùng CPU, không chạy GPU/Modal).
Input: OOF/test probs của 14 full-OOF runs trong `outputs/runs/*`, `outputs/train_labels.npy`, `outputs/train_langs.npy`, độ dài text (chỉ lấy `len`, đọc từ zip trong bộ nhớ; không in, không gửi text đi đâu).
Scripts (scratchpad): `C:\Users\Admin\AppData\Local\Temp\claude\d--A-RMITHACKATHON\1170d8c9-19f8-4773-beaf-7cf8b7cc5e2a\scratchpad\team\decision\` (`common.py`, `methods.py`, `run_nested.py`, `table.py`, `diversity.py`, `err_pool.py`, `bj_probe.py`, `slice_probe.py`).

## TL;DR

- **Không có decision layer nào thắng blend9 equal mean ≥ +0.0005 trong nested CV.** Vì vậy **không tạo** `outputs/runs/decision_best/`. Nên giữ **blend9 equal mean** (OOF 0.9770) làm bài nộp chính.
- Tốt nhất là **B/J tie-break bằng vote của 14 models**: trong vùng blend9 lưỡng lự giữa BENIGN và JAILBREAK (|pB − pJ| < 0.2), dự đoán JAILBREAK nếu ≥ 8/14 models vote JB. Kết quả là **+0.0002** (trung bình 10 outer seeds; dương ở 6 seeds, âm ở 3). Trên OOF nó chỉ đổi 41 dòng (24 đúng thêm / 17 sai thêm), trên test 15 dòng. Mức này là noise, nên không đáng dùng một suất final selection.
- Các method có **fit weights/params** đều **thua** blend9: Caruana −0.0010…−0.0021, BMC −0.0014…−0.0026, supervised Dawid–Skene (inner CV) −0.0009, cascade −0.0007…−0.0009, temperature equalisation −0.0005…−0.0010, HGB stacking −0.0007/−0.0008. Kết quả này khớp với các lần reject trước đó (fitted weights, LR stacking, multipliers).
- Slice routing (pt-long vs phần còn lại) có vẻ +0.0005 trên 3 seeds, nhưng **về 0.0000 / −0.0004 khi chạy 10 seeds**. Lựa chọn candidate thay đổi theo từng fold, nên đây là selection noise.
- **Lý do không còn headroom**: các members rất tương quan với nhau. Disagreement trung bình giữa hai models là 3.1%, error Jaccard 0.33. **Mọi lỗi của blend9 đều có ≥ 4/9 members sai.** 199/286 lỗi nằm ở pt-long, và confusion ở đó **đối xứng** (BEN→JB 102, JB→BEN 97), nên không threshold hay bias nào sửa được. Candidate tốt nhất trên pt-long chỉ có 196 lỗi, so với 199 của blend9.
- **Selection optimism của blend9**: chạy lại đúng quy trình đã tạo ra blend9 (forward selection, equal weights) dưới dạng nested thì được **0.9764**. Như vậy 0.9770 đã chứa khoảng 0.0006 optimism, chưa tính phần optimism từ pseudo-label OOF. Kỳ vọng public của blend9 ≈ 0.973 ± 0.003 (tham chiếu lần nộp trước: OOF 0.9666 → public 0.9635).

## Protocol

- Outer: `StratifiedKFold(5, shuffle=True, random_state=7)` theo `label_language`. Mọi lựa chọn (subset, weight, threshold, rule) đều làm **chỉ trên outer-train** (11,200 dòng). Score là macro-F1 pooled trên 14,000 dự đoán outer-test.
- Hyper-parameter của procedure (ví dụ threshold `t` của cascade, `K`/objective của Caruana, slicer/min_gain của routing, tau của DS) được chọn bằng **inner CV** 5-fold (seed 11) bên trong outer-train, rồi refit trên toàn bộ outer-train.
- Baseline blend9 không có tham số nào được fit, nên nested = non-nested = **0.9770** (pt 0.9690; độ lệch chuẩn giữa các fold 0.0034).
- Đo độ tin cậy: wins/losses so với blend9 trên từng dòng, bootstrap 2,000 lần cho delta (seed 7), và lặp lại với outer seeds 7–9 (mọi method) hoặc 7–16 (các method đứng đầu).
- Pool gồm 14 runs. Lưu ý: OOF của `mmb_pseudo_ep4` và `wolf_pseudo_ep4` hơi optimistic, vì teacher đã thấy val folds qua test pseudo-labels.

## Kết quả (sắp xếp theo delta; dùng số 10 seeds nếu có)

| method | seeds | nested macro-F1 (mean) | delta vs blend9 | seed-7 wins/losses | seed-7 bootstrap P(delta>0) | seed-7 95% CI |
|---|---|---|---|---|---|---|
| e_plugin_f1_raw | 3 | 0.9773 | +0.0003 | 3/3 | 0.82 | [-0.0002, +0.0010] |
| d_bj_tiebreak_votes_only | 10 | 0.9771 | +0.0002 | 24/17 | 0.86 | [-0.0003, +0.0011] |
| blend9_mean (baseline) | 3 | 0.9770 | +0.0000 | 0/0 | n/a | n/a |
| e_entropy_weighted | 3 | 0.9770 | +0.0000 | 0/0 (chọn beta=0) | n/a | n/a |
| e_trimmed_b9 | 3 | 0.9770 | +0.0000 | 0/0 | n/a | n/a |
| a_slice_pt_long_means | 10 | 0.9770 | -0.0000 | 28/28 | 0.62 | [-0.0014, +0.0018] |
| e_median_b9 | 3 | 0.9770 | -0.0000 | 11/16 | 0.49 | [-0.0010, +0.0011] |
| e_ds_sup_soft_b9 | 10 | 0.9769 | -0.0000 | 14/19 | 0.38 | [-0.0020, +0.0015] |
| d_bj_tiebreak (rule pool đầy đủ) | 10 | 0.9769 | -0.0001 | 24/17 | 0.86 | [-0.0003, +0.0011] |
| a_slice_pt3 | 3 | 0.9767 | -0.0003 | 26/30 | 0.32 | [-0.0020, +0.0012] |
| e_ds_em_soft_b9 (unsupervised DS) | 10 | 0.9767 | -0.0003 | 18/23 | 0.51 | [-0.0016, +0.0017] |
| a_slice_pt_long | 10 | 0.9766 | -0.0004 | 28/29 | 0.52 | [-0.0017, +0.0016] |
| e_ds_sup_hard_b9 | 3 | 0.9766 | -0.0004 | 16/24 | 0.30 | [-0.0022, +0.0012] |
| e_power_mean | 3 | 0.9766 | -0.0004 | 1/3 | 0.07 | [-0.0010, +0.0001] |
| e_plugin_f1_cal | 3 | 0.9765 | -0.0005 | 3/9 | 0.15 | [-0.0014, +0.0004] |
| e_ds_em_hard_all14 | 3 | 0.9765 | -0.0005 | 24/32 | 0.24 | [-0.0022, +0.0010] |
| e_tempeq_all14 | 3 | 0.9765 | -0.0005 | 26/31 | 0.23 | [-0.0020, +0.0008] |
| all14_mean | 3 | 0.9764 | -0.0006 | 24/30 | 0.22 | [-0.0021, +0.0009] |
| c_forward_noreplace_f1 (= quy trình tạo blend9) | 3 | 0.9764 | -0.0006 | 24/30 | 0.22 | [-0.0021, +0.0009] |
| e_trimmed_all14 | 3 | 0.9764 | -0.0006 | 24/30 | 0.22 | [-0.0021, +0.0009] |
| a_slice_pt_long_core8_mg | 3 | 0.9764 | -0.0006 | 0/6 | 0.00 | [-0.0021, -0.0003] |
| blend7_mean | 3 | 0.9764 | -0.0006 | 20/33 | 0.17 | [-0.0019, +0.0007] |
| a_slice_innercv | 3 | 0.9764 | -0.0006 | 7/13 | 0.00 | [-0.0026, -0.0003] |
| f_hgb_stack_nometa | 3 | 0.9763 | -0.0007 | 39/42 | 0.40 | [-0.0022, +0.0016] |
| b_cascade_t0.75 | 3 | 0.9762 | -0.0007 | 5/10 | 0.02 | [-0.0018, -0.0000] |
| b_cascade_t0.9 | 3 | 0.9762 | -0.0007 | 5/10 | 0.02 | [-0.0018, -0.0000] |
| a_slice_lang_long | 3 | 0.9762 | -0.0008 | 34/36 | 0.51 | [-0.0020, +0.0019] |
| f_hgb_stack_meta (+len, lang) | 3 | 0.9762 | -0.0008 | 37/42 | 0.39 | [-0.0022, +0.0016] |
| b_cascade_innercv | 3 | 0.9761 | -0.0009 | 8/15 | 0.02 | [-0.0020, -0.0001] |
| e_ds_sup_innercv | 3 | 0.9761 | -0.0009 | 9/26 | 0.03 | [-0.0031, +0.0000] |
| e_tempeq_b9 | 3 | 0.9760 | -0.0010 | 9/14 | 0.01 | [-0.0021, -0.0001] |
| c_caruana_innercv | 3 | 0.9759 | -0.0010 | 19/29 | 0.12 | [-0.0023, +0.0005] |
| a_slice_pt_long_core8 | 3 | 0.9758 | -0.0012 | 2/12 | 0.00 | [-0.0026, -0.0004] |
| e_bmc_b9_tempered | 3 | 0.9756 | -0.0014 | 7/17 | 0.01 | [-0.0027, -0.0003] |
| e_bmc_all14_tempered | 3 | 0.9756 | -0.0014 | 18/30 | 0.04 | [-0.0026, +0.0001] |
| c_caruana_f1_b9init_K10 | 3 | 0.9755 | -0.0015 | 20/26 | 0.11 | [-0.0023, +0.0005] |
| c_caruana_nll_best1_K20 | 3 | 0.9754 | -0.0016 | 27/41 | 0.02 | [-0.0035, -0.0000] |
| c_caruana_bag20_f1_K15 | 3 | 0.9753 | -0.0016 | 20/33 | 0.02 | [-0.0031, -0.0001] |
| c_caruana_f1_best1_K20 | 3 | 0.9749 | -0.0021 | 19/33 | 0.03 | [-0.0028, +0.0001] |
| e_bmc_b9 | 3 | 0.9744 | -0.0026 | 18/31 | 0.02 | [-0.0032, -0.0001] |

Delta theo từng seed (10 seeds) của các method đứng đầu:
- `d_bj_tiebreak_votes_only`: +4,+4,+4,+4,−1,0,−2,+4,+4,−3 (×1e-4).
- `a_slice_pt_long_means`: +3,+9,+3,+2,−9,+1,−7,+3,−3,−3.
- `a_slice_pt_long`: +1,+9,+1,+2,−13,−12,−9,−7,−6,−7.

## Các method đã thử

- **(a) Slice routing.** Slicers: pt×{≤400,>400}; lang×{≤400,>400}; pt×{≤400, 400–1600, >1600}+other. Với mỗi slice, chọn 1 trong 47 candidates (blend9/7/8, all14, family means, 14 single models, 9 lần leave-one-out từ blend9). Phép chọn là coordinate ascent trên **global** macro-F1 của outer-train. Có thêm các biến thể: library chỉ gồm means, library core-8, min_gain, inner CV. In-sample tăng từ 0.9770 lên 0.9786, nhưng nested chỉ còn ≈ 0. Lựa chọn trên slice pt-long đổi qua lại giữa các fold (b9 minus wolf_ep4 / minus wolf_ep3 / b9).
- **(b) Cascade.** Nếu max-prob (hoặc margin) của blend9 ≥ t thì giữ blend9; nếu không thì giao cho specialist (chọn từ library, fit trên các dòng uncertain của outer-train), có tùy chọn chỉ áp dụng cho pt-long. `t` chọn bằng inner CV. Kết quả −0.0007…−0.0009. Các dòng uncertain (conf < 0.9: 1,047 dòng, acc 0.79) không có candidate nào tốt hơn blend9 một cách ổn định.
- **(c) Caruana ensemble selection** (with replacement; init best-1 hoặc blend9; objective F1+NLL tie-break hoặc NLL; bagging 20×50%; K chọn bằng inner CV). Mọi biến thể đều −0.0010…−0.0021. Fitted weights overfit, equal mean vẫn tốt nhất.
- **(d) Disagreement-aware B/J tie-break.** Trong vùng top-2 = {B, J} và |pB − pJ| < m: dùng vote9/vote14 với ngưỡng q, mean của all14/pseudo2/family, hoặc một single model. Grid được fit trên outer-train. Với rule pool chỉ gồm votes, cả 5 folds × 3 seeds đều chọn (m=0.2, vote14, q=8), và trên full data cũng ra đúng rule này. Tuy vậy landscape có dạng spike (q=7: −0.0004, q=9: 0.0000), và 10 seeds chỉ cho +0.0002.
- **(e) Các rule có nguyên lý khác.**
  - Supervised Dawid–Skene: mỗi model là một noisy annotator với confusion matrix fit trên outer-train (hard/soft, tau correlation-tempering).
  - Unsupervised DS–EM: transductive, không dùng label.
  - Temperature equalisation: temperature riêng cho từng model rồi mới average.
  - Entropy-weighted mean và power mean (q fit).
  - Median / trimmed mean.
  - Bayesian Model Combination (Monteith et al. 2011; có biến thể tempered).
  - Plug-in expected-macro-F1 decoding (Jansche 2007 / Ye et al. 2012; label-free, có hoặc không calibration).
  - HistGradientBoosting stacker trên probs, disagreement std, votes, log-len, lang.

  Không có biến thể nào vượt ngưỡng +0.0005. Plug-in decoding không cần fit gì, nhưng chỉ đổi 6 dòng OOF và 1 dòng test, nên không có tác dụng thực tế.

## Vì sao không còn headroom ở tầng decision

| thống kê | giá trị |
|---|---|
| blend9 OOF errors | 286 (pt-long 199, pt-short 20, other-long 26, other-short 41) |
| Errors theo số members sai (trên 9) | ≤3: **0**, 4: 11, 5: 62, 6: 58, 7: 45, 8: 51, 9: 59 |
| Dòng mà đa số members sai nhưng blend9 vẫn đúng | 22 |
| Oracle any-of-14 errors (sàn không sửa được) | 43 |
| Errors có blend9 conf ≥ 0.9 | 62 (41 trong số đó cả 14 models đều sai, nhiều khả năng là label noise) |
| Disagreement trung bình giữa hai members | 3.1% dòng |
| Error Jaccard giữa hai members | trung bình 0.33 (min 0.25, max 0.53); giữa các family 0.27–0.35 |
| pt-long confusion (true B / J) | B→J 102, J→B 97: đối xứng, nên không có bias nào để chỉnh |
| Test conf < 0.9 / OOF conf < 0.9 | 7.3% / 7.5%: phân phối confidence giống nhau, nên threshold rules có chuyển sang test được (chỉ là không có gain) |

Một decision layer chỉ sửa được lỗi khi có một tín hiệu (model, family hay slice) **đúng một cách ổn định** ở chỗ blend9 sai. Ở đây không có tín hiệu như vậy: mọi lỗi đều có ≥ 4/9 members cùng sai, và trên slice nặng nhất (pt-long), candidate tốt nhất chỉ hơn blend9 3 dòng in-sample. Mức noise giữa các outer seeds của một procedure có chọn lựa là khoảng ±0.0007 (≈ ±10 dòng), lớn hơn mọi gain đo được.

## Khuyến nghị cho team lead

1. **Final selection:** giữ **blend9 equal mean** (hoặc biến thể của nó có thêm full-data refits nếu team quyết định như vậy; phần này không thuộc phạm vi decision layer và không có OOF để kiểm chứng). Không dùng suất nộp cho decision layers. Nếu vẫn muốn dùng, vote14 tie-break chỉ đổi 15 dòng test; kỳ vọng ≈ +0.0002, tức là noise.
2. Các blend7/8/9 chỉ khác nhau 12–16 dòng test. Chọn cả 3 làm final thì gần như là một bài nộp lặp lại 3 lần, nên dùng các suất final cho những hướng có tín hiệu thật sự khác nhau.
3. Muốn rút ngắn gap tới 0.99 thì cần **tín hiệu mới trên pt-long BENIGN↔JAILBREAK** (hiểu harm semantics của role-play kiểu WildJailbreak), không phải một tầng decision mới. Một member mới chỉ đáng thêm vào nếu làm giảm khoảng ≥ 10 lỗi pt-long trong nested addition test. Thấp hơn mức đó thì không phân biệt được với noise.
4. Khi báo kỳ vọng LB, hãy trừ khoảng 0.0006 selection optimism (blend9 được chọn trên full OOF) và optimism từ pseudo OOF, rồi trừ thêm khoảng −0.003 OOF→public gap đã quan sát được.
