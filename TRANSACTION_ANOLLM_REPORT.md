# Báo Cáo Train AnoLLM Cho Dataset Transaction

## 1. Mục Tiêu

Mục tiêu của thực nghiệm này là áp dụng AnoLLM cho bài toán phát hiện giao dịch bất thường trên dữ liệu transaction giả lập.

AnoLLM không train một classifier supervised để dự đoán trực tiếp `fraud` hay `normal`. Thay vào đó, model học phân phối của các giao dịch bình thường. Khi gặp một giao dịch mới, nếu giao dịch đó khác nhiều so với pattern normal đã học, model sẽ gán anomaly score cao hơn.

## 2. Dataset

Dataset được lưu tại:

```text
data/transaction/transaction.csv
```

Tổng số dòng dữ liệu:

```text
1110 rows
```

Phân bố label:

```text
Normal, is_anomaly = 0: 1080 rows
Fraud,  is_anomaly = 1:   30 rows
```

Tỉ lệ fraud trên toàn bộ dataset:

```text
30 / 1110 = 2.70%
```

Danh sách cột:

```text
transaction_id,user_id,amount,currency,merchant,category,country,device,hour,payment_method,is_anomaly
```

Trong đó:

```text
is_anomaly = 0: giao dịch bình thường
is_anomaly = 1: giao dịch fraud / bất thường
```

Fraud trong dataset giả lập được tạo bằng các dấu hiệu như:

- amount quá lớn
- giao dịch lúc nửa đêm
- quốc gia lạ
- merchant lạ
- device lạ
- nhiều dấu hiệu bất thường xuất hiện cùng lúc

## 3. Cách Chia Train/Test

Thực nghiệm sử dụng setting:

```text
semi_supervised
train_ratio = 0.75
n_splits = 1
split_idx = 0
```

Kết quả split:

```text
Train set: 810 rows
  Normal: 810
  Fraud:    0

Test set: 300 rows
  Normal: 270
  Fraud:   30
```

Tỉ lệ fraud trong test set:

```text
30 / 300 = 10%
```

Trong setting `semi_supervised`, tập train chỉ gồm normal transactions. Tập test gồm normal còn lại và toàn bộ fraud transactions.

## 4. Lý Do Dùng Semi-Supervised

Trong thực tế, giao dịch fraud rất đa dạng và thay đổi liên tục. Hacker hoặc attacker luôn thay đổi pattern để tránh bị phát hiện.

Nếu train supervised, model học trực tiếp từ các label fraud đã biết. Cách này có điểm mạnh khi fraud pattern trong train và test giống nhau, nhưng có rủi ro là model chỉ nhận ra những kiểu fraud đã từng thấy.

Semi-supervised thì khác. Model chỉ cần học giao dịch bình thường trông như thế nào. Khi một giao dịch mới khác nhiều với normal pattern, model có thể gán anomaly score cao, kể cả khi kiểu fraud đó chưa từng xuất hiện trong train.

Nói ngắn gọn:

```text
Supervised:
  Học fraud đã biết -> tốt với known fraud, yếu hơn với fraud mới.

Semi-supervised:
  Học normal behavior -> phát hiện giao dịch lệch khỏi normal, kể cả unknown fraud.
```

Đây là lý do semi-supervised phù hợp với anomaly detection trong giao dịch tài chính, nơi fraud pattern có thể thay đổi liên tục.

## 5. Mô Hình Và Thông Số Train

Model LLM sử dụng:

```text
distilgpt2
```

Đường dẫn model sau train:

```text
exp/transaction/semi_supervised/split1/split0_large_bs4_steps500/models/anollm_lr5e-05_standard_distilgpt2_test.pt
```

Thông số train:

```text
batch_size = 4
max_steps = 500
eval_steps = 100
learning_rate = 5e-5
binning = standard
setting = semi_supervised
```

Kết quả train:

```text
train_loss = 0.7306
epoch = 2.46
train_runtime = 1572.33 seconds
```

Với dataset mới, model chỉ đi qua train set khoảng 2.46 epochs, hợp lý hơn so với lần trước dataset quá nhỏ khiến epoch lên hơn 16 và dễ overfit.

## 6. Kiến Trúc Tổng Thể

Luồng xử lý chính:

```text
transaction.csv
  -> src/data_utils.py
  -> split train/test
  -> anollm/anollm_dataset.py
  -> serialize tabular row thành text
  -> tokenizer của distilgpt2
  -> anollm/anollm.py
  -> Hugging Face Trainer
  -> fine-tune causal language model
  -> save model
  -> evaluate_anollm.py
  -> anomaly scores
```

Mỗi dòng tabular được biến thành text dạng:

```text
amount is 120.5, country is VN, device is ios, merchant is FreshMart, ...
```

Sau đó LLM được fine-tune bằng causal language modeling, tức là dự đoán token tiếp theo dựa trên các token trước đó.

AnoLLM không học label `is_anomaly` trong train. Label chỉ được dùng để chia dữ liệu và đánh giá sau train.

## 7. Cách Tính Anomaly Score

Sau khi train, với mỗi transaction trong test set, model tính negative log-likelihood / loss khi đọc chuỗi text của transaction đó.

Ý tưởng:

```text
Transaction giống normal:
  model dự đoán dễ hơn
  loss thấp hơn
  anomaly_score thấp hơn

Transaction bất thường:
  model dự đoán khó hơn
  loss cao hơn
  anomaly_score cao hơn
```

Do đó:

```text
anomaly_score càng cao -> giao dịch càng bất thường
```

File score kèm label thật:

```text
exp/transaction/semi_supervised/split1/split0_large_bs4_steps500/scores/transaction_scores_with_labels.csv
```

Format:

```text
row_index,anomaly_score,is_anomaly
```

## 8. Kết Quả Evaluate

Số dòng evaluate:

```text
300 rows
```

Phân bố trong test:

```text
Normal: 270
Fraud:   30
```

Thống kê anomaly score:

```text
Normal:
  mean = 43.91
  min  = 38.90
  max  = 54.07

Fraud:
  mean = 87.18
  min  = 57.67
  max  = 174.15
```

Chỉ số trên synthetic dataset:

```text
ROC-AUC = 1.0
Average Precision = 1.0
```

Lưu ý: kết quả này đạt cao vì dataset là dữ liệu giả lập và fraud pattern được tạo khá tách biệt với normal. Với dữ liệu thật, cần đánh giá lại bằng tập test thực tế, pattern fraud phức tạp hơn và có nhiều trường hợp gần ranh giới hơn.

## 9. Lệnh Đã Sử Dụng

Generate dataset:

```powershell
conda run -n anollm python scripts/generate_transaction_dataset.py
```

Train:

```powershell
python train_anollm.py --dataset transaction --data_dir data --n_splits 1 --split_idx 0 --train_ratio 0.75 --setting semi_supervised --binning standard --model distilgpt2 --batch_size 4 --max_steps 500 --eval_steps 100 --exp_dir exp/transaction/semi_supervised/split1/split0_large_bs4_steps500
```

Evaluate:

```powershell
python evaluate_anollm.py --dataset transaction --data_dir data --n_splits 1 --split_idx 0 --train_ratio 0.75 --setting semi_supervised --binning standard --model distilgpt2 --batch_size 8 --n_permutations 1 --exp_dir exp/transaction/semi_supervised/split1/split0_large_bs4_steps500
```

Export score kèm label:

```powershell
python scripts/export_scores_with_labels.py --train_ratio 0.75 --exp_dir exp/transaction/semi_supervised/split1/split0_large_bs4_steps500
```

## 10. Kết Luận

Thực nghiệm cho thấy AnoLLM có thể áp dụng cho dataset transaction custom bằng cách serialize mỗi dòng giao dịch thành text và fine-tune LLM trên normal transactions.

Với dataset mới lớn hơn, train set có 810 normal rows và model chỉ train khoảng 2.46 epochs, giảm nguy cơ overfit so với bản dataset nhỏ trước đó. Score evaluate cho thấy fraud transactions có anomaly score cao hơn normal transactions rõ rệt trên tập dữ liệu giả lập.
