# Báo Cáo Train AnoLLM Cho Dataset Transaction

## 0. Tổng Quan Thí Nghiệm

Thí nghiệm đánh giá AnoLLM trên một dataset transaction giả lập được thiết kế để bài toán có độ khó thực tế (fraud không tách biệt hoàn toàn khỏi normal):

- **Dataset**: 3300 dòng, 300 fraud, gồm nhiều loại fraud "borderline" chồng lấn với normal.
- **Đánh giá chặt chẽ**: bootstrap 95% CI, phương sai qua 16 permutation, ablation số permutation, phân tích threshold, recall theo từng loại fraud.
- **Explainability**: tính anomaly score theo từng feature để chỉ ra feature nào khiến model gắn cờ một giao dịch là fraud.

> Tất cả chạy trên **CPU-only**. Train ~37 phút, evaluate (16 permutations) ~46 phút.

---

## 1. Mục Tiêu

Áp dụng AnoLLM cho bài toán phát hiện giao dịch bất thường. AnoLLM **không** train classifier supervised để dự đoán trực tiếp `fraud`/`normal`. Model học phân phối của giao dịch bình thường; giao dịch nào lệch nhiều khỏi pattern normal sẽ nhận anomaly score (negative log-likelihood) cao hơn.

## 2. Dataset

Lưu tại `data/transaction/transaction.csv`. Sinh bằng `scripts/generate_transaction_dataset.py`.

```text
Tổng:   3300 rows
Normal (is_anomaly=0): 3000
Fraud  (is_anomaly=1):  300   (9.09%)
```

Cột: `transaction_id, user_id, amount, currency, merchant, category, country, device, hour, payment_method, is_anomaly`.

### 2.1. Phân bố dataset

- **Nâng đuôi phân phối normal**: bỏ cap cứng 320, `amount` normal theo lognormal có đuôi dày (sigma 0.7). Khoảng 3% normal xảy ra ban đêm, 2% normal đến từ country hiếm → các dấu hiệu "ban đêm" hay "country lạ" **không còn là chỉ báo hoàn hảo**.
- **Thêm các loại fraud borderline** chồng lấn với normal, bên cạnh vài loại tách biệt giữ lại cho đa dạng.

Phân bố theo loại (lưu ở sidecar `data/transaction/transaction_meta.csv`, **không** đưa vào model):

```text
subtle_amount                68   amount 250-600 (chồng lấn đuôi normal)
odd_hour_local               61   country bình thường + giờ 1-4h + amount hơi cao
merchant_category_mismatch   47   merchant quen + category lệch ngữ cảnh (crypto/luxury/...)
rare_country                 34   country hiếm
velocity_like                32   gần như normal: amount 150-350 + giờ 5/6/23 (cố tình rất khó)
large_amount                 26   amount 800-3000
rare_device                  20   device lạ + payment lạ
combined                     12   nhiều tín hiệu mạnh cùng lúc (giữ dễ)
```

### 2.2. Vùng chồng lấn giữa normal và fraud

```text
Normal amount  p50 / p95 / max : 36.6 / 123.1 / 433.3
Fraud  amount  min / p50 / p95 : 10.5 / 365.3 / 2484.9
```

Đuôi normal (tới ~433) chồng lấn trực tiếp với vùng `subtle_amount` (250–600) và một phần `large_amount`, nên `amount` không còn tách normal/fraud một cách tuyệt đối.

`transaction_meta.csv` chỉ phục vụ phân tích/ablation; `transaction.csv` giữ nguyên 11 cột như cũ để không rò rỉ nhãn cho model.

## 3. Cách Chia Train/Test

```text
setting     = semi_supervised
train_ratio = 0.75
n_splits    = 1, split_idx = 0
```

```text
Train set: 2250 rows   (Normal: 2250, Fraud: 0)
Test  set: 1050 rows   (Normal:  750, Fraud: 300)
Tỉ lệ fraud trong test: 300/1050 = 28.6%
```

Trong `semi_supervised`, train chỉ gồm normal; test gồm normal còn lại + toàn bộ fraud.

## 4. Lý Do Dùng Semi-Supervised

Fraud thực tế đa dạng và thay đổi liên tục. Supervised học fraud đã biết → tốt với known fraud nhưng yếu với fraud mới. Semi-supervised chỉ học "normal trông như thế nào", nên có thể phát hiện cả fraud chưa từng thấy khi nó lệch khỏi normal. Đây là lý do semi-supervised phù hợp với anomaly detection trong giao dịch tài chính.

## 5. Mô Hình Và Thông Số Train

```text
model        = distilgpt2
binning      = standard
batch_size   = 4
max_steps    = 800
eval_steps   = 200
learning_rate= 5e-5
setting      = semi_supervised
```

Kết quả train:

```text
train_loss    = 0.6899
epoch         = 1.42      (đi qua train set ~1.42 lần)
train_runtime = 2230 s    (~37 phút, CPU)
```

Model lưu tại:

```text
exp/transaction/semi_supervised/split1/split0_hard/models/anollm_lr5e-05_standard_distilgpt2_test.pt
```

## 6. Kiến Trúc Tổng Thể

```text
transaction.csv
  -> src/data_utils.py (split train/test, binning standard)
  -> anollm/anollm_dataset.py (serialize mỗi dòng thành text)
  -> tokenizer distilgpt2
  -> anollm/anollm.py + HuggingFace Trainer (fine-tune causal LM trên normal)
  -> evaluate_anollm.py (anomaly score = NLL)
```

Mỗi dòng được serialize dạng: `amount is ..., country is VN, device is ios, ...`. Model fine-tune bằng causal language modeling. Nhãn `is_anomaly` **không** dùng khi train, chỉ dùng để chia dữ liệu và đánh giá.

## 7. Cách Tính Anomaly Score

Với mỗi transaction, model tính negative log-likelihood (loss) của chuỗi text. Giống normal → loss thấp → score thấp; bất thường → loss cao → score cao. Để giảm nhiễu do thứ tự cột, score là trung bình qua nhiều **permutation** thứ tự cột (ở đây 16).

File score kèm nhãn: `.../scores/transaction_scores_with_labels.csv` (`row_index, anomaly_score, is_anomaly`).

## 8. Kết Quả Evaluate (đã làm chặt chẽ)

Số dòng evaluate: 1050 (Normal 750, Fraud 300). `n_permutations = 16`, inference ~2790 s (~46 phút, CPU).

### 8.1. Thống kê anomaly score

```text
Normal: mean 43.73,  range 38.61 - 51.90
Fraud : mean 67.93,  range 48.53 - 174.24
Vùng chồng lấn: normal p95 = 48.13  vs  fraud p05 = 51.91
```

**Có vùng chồng lấn** giữa normal và fraud (normal max 51.90 > fraud min 48.53) → đây là nơi phát sinh các ca sai.

### 8.2. Metric chính kèm khoảng tin cậy (bootstrap 1000 lần)

| Metric            | Điểm   | Bootstrap 95% CI |
| ----------------- | ------ | ---------------- |
| ROC-AUC           | 0.9995 | 0.9990 – 0.9998  |
| Average Precision | 0.9988 | 0.9975 – 0.9995  |
| F1@k              | 0.9800 | 0.9649 – 0.9903  |

(F1@k: chọn top-k điểm cao nhất làm dương, k = số fraud thật; tại điểm này Precision = Recall = 0.9800.)

### 8.3. Độ ổn định qua 16 permutation

```text
ROC-AUC : 0.9993 +/- 0.0002   [0.9986, 0.9996]
F1@k    : 0.9794 +/- 0.0021
```

Kết quả **rất ổn định**, không phụ thuộc một lần chạy may rủi.

### 8.4. Ablation số permutation (gộp điểm trung bình)

| n_perm | ROC-AUC | AP     | F1@k   |
| ------ | ------- | ------ | ------ |
| 1      | 0.9993  | 0.9982 | 0.9767 |
| 2      | 0.9994  | 0.9984 | 0.9767 |
| 4      | 0.9995  | 0.9988 | 0.9800 |
| 8      | 0.9995  | 0.9988 | 0.9800 |
| 16     | 0.9995  | 0.9988 | 0.9800 |

→ Từ ~4 permutation trở lên là đủ; tăng thêm không cải thiện đáng kể.

### 8.5. Phân tích threshold

```text
best-F1            : 0.9816  (P=0.9833, R=0.9800)
F1 @ contamination : 0.9800  (P=0.9800, R=0.9800)   # ngưỡng = top 28.6% điểm cao nhất
```

### 8.6. Recall theo từng loại fraud

Ngưỡng = top 28.6% điểm cao nhất:

| anomaly_type               | n   | caught | recall    |
| -------------------------- | --- | ------ | --------- |
| combined                   | 12  | 12     | 1.000     |
| large_amount               | 26  | 26     | 1.000     |
| merchant_category_mismatch | 47  | 47     | 1.000     |
| odd_hour_local             | 61  | 61     | 1.000     |
| rare_country               | 34  | 34     | 1.000     |
| rare_device                | 20  | 20     | 1.000     |
| **subtle_amount**          | 68  | 65     | **0.956** |
| **velocity_like**          | 32  | 29     | **0.906** |
| **ALL FRAUD**              | 300 | 294    | **0.980** |

Các loại tách biệt → recall 100%. Hai loại **borderline** chồng lấn normal (`subtle_amount`, `velocity_like`) bị bỏ sót một phần — cho thấy bài toán có độ khó thực sự.

## 9. Phân Tích Giải Thích (Explainability): Vì Sao LLM Gắn Cờ?

Dùng `decision_function(feature_wise=True)` để tách NLL theo **từng feature** (`scripts/explain_anomalies.py`, 4 permutation). Với mỗi feature, chuẩn hóa theo baseline normal:

```text
z_feature(row) = (nll_feature(row) - mean_normal) / std_normal
```

Feature có z cao nhất = lý do chính khiến giao dịch bị gắn cờ.

### 9.1. Đóng góp trung bình mỗi feature (nhóm fraud)

```text
device          70.41      hour            1.30
payment method  45.69      country         1.17
category         7.97      currency        0.05
merchant         4.99      user id        -0.01
amount           4.55
transaction id   4.60   (cột ID, xem lưu ý 9.3)
```

### 9.2. Feature dẫn dắt theo từng loại fraud (đã bỏ 2 cột ID)

```text
combined                    -> device         (100%)
rare_device                 -> device         (100%)
large_amount                -> amount         (100%)
subtle_amount               -> amount         (100%)
rare_country                -> country         (70%)
odd_hour_local              -> amount          (63%)
velocity_like               -> hour            (56%)
merchant_category_mismatch  -> payment method  (53%)
```

Model "nhìn" đúng dấu hiệu: `subtle_amount`/`large_amount` → `amount`; `rare_country` → `country`; `velocity_like` → `hour`; `rare_device`/`combined` → `device`. Đây là bằng chứng trực tiếp cho câu hỏi "vì sao là fraud".

### 9.3. Ví dụ giao dịch cụ thể (score cao nhất)

```text
TX003041 [combined] -> device(z=888.0), payment method(z=264.6), merchant(z=147.5)
TX003241 [combined] -> device(z=899.5), payment method(z=190.8), merchant(z=146.4)
TX003158 [combined] -> device(z=884.3), payment method(z=265.2), merchant(z=109.2)
```

**Lưu ý (limitation):** `transaction_id` (và `user_id`) là cột ID, gần như duy nhất nên NLL nhiễu. Với các fraud borderline không có feature nào thật sự bất thường, `transaction_id` đôi khi trở thành "feature dẫn dắt" với z thấp (~6) — chính là dấu hiệu model **không** có tín hiệu mạnh, và đây cũng là các ca dễ bị bỏ sót. Khuyến nghị: loại bỏ cột ID khỏi input ở các thí nghiệm sau.

File chi tiết: `.../scores/feature_attribution.csv`. Biểu đồ: `roc_curve.png`, `pr_curve.png`, `score_hist.png`. Toàn bộ số liệu: `.../scores/robust_metrics.json`.

## 10. Lệnh Đã Sử Dụng

Generate dataset (lớn + khó):

```powershell
conda run -n anollm python scripts/generate_transaction_dataset.py --n_normal 3000 --n_anomaly 300
```

Train:

```powershell
python train_anollm.py --dataset transaction --data_dir data --n_splits 1 --split_idx 0 --train_ratio 0.75 --setting semi_supervised --binning standard --model distilgpt2 --batch_size 4 --max_steps 800 --eval_steps 200 --exp_dir exp/transaction/semi_supervised/split1/split0_hard
```

Evaluate (16 permutation):

```powershell
python evaluate_anollm.py --dataset transaction --data_dir data --n_splits 1 --split_idx 0 --train_ratio 0.75 --setting semi_supervised --binning standard --model distilgpt2 --batch_size 8 --n_permutations 16 --exp_dir exp/transaction/semi_supervised/split1/split0_hard
```

Export score + đánh giá robust + explainability + recall theo loại:

```powershell
python scripts/export_scores_with_labels.py --train_ratio 0.75 --exp_dir exp/transaction/semi_supervised/split1/split0_hard
python scripts/evaluate_robust.py --exp_dir exp/transaction/semi_supervised/split1/split0_hard --train_ratio 0.75
python scripts/explain_anomalies.py --exp_dir exp/transaction/semi_supervised/split1/split0_hard --train_ratio 0.75 --n_permutations 4 --batch_size 16
python scripts/per_type_recall.py --exp_dir exp/transaction/semi_supervised/split1/split0_hard
```

## 11. Kết Luận

- Trên dataset 3300 dòng, 300 fraud (nhiều loại borderline chồng lấn normal), AnoLLM phân biệt fraud rất tốt: ROC-AUC 0.9995, AP 0.9988, F1@k 0.98. Kết quả đi kèm **khoảng tin cậy hẹp** (bootstrap) cùng **độ ổn định cao** qua 16 permutation — tức là đáng tin chứ không phải may rủi một lần chạy.
- **Bài toán đã thực sự khó**: hai loại fraud borderline (`subtle_amount` 95.6%, `velocity_like` 90.6%) bị bỏ sót một phần, và score normal/fraud đã có vùng chồng lấn. 6/300 fraud bị miss rơi đúng vào nhóm này.
- **Đã trả lời được "vì sao"**: phân tích feature-wise cho thấy model gắn cờ dựa trên đúng feature bất thường của từng loại (amount/country/hour/device/payment), và các ca model "đuối" là các ca không có feature nào thật sự lệch.
- **Hướng tiếp theo**: bỏ cột ID (`transaction_id`, `user_id`) khỏi input; thử nhiều seed split để có CI theo cả phương sai dữ liệu; tăng tỉ trọng các loại borderline hoặc thêm nhiễu để hạ trần metric; thử model lớn hơn hoặc binning khác (ablation cần train lại).
