# AnoLLM Transaction Admin Frontend

MVP frontend Tailwind cho pipeline AnoLLM transaction.

## Tabs

- Dashboard: KPI, score separation, Precision/Recall/F1 theo threshold NLL.
- Dataset: đọc dữ liệu thật từ `data/transaction/transaction.csv`, có search, filter label, pagination.
- Training: form cấu hình train, tạo/copy command, nút Run Training để nối backend sau này.
- Evaluation: form cấu hình evaluate, threshold simulator, tạo/copy command, nút Run Evaluate để nối backend sau này.
- Scores: đọc score thật từ `transaction_scores_with_labels.csv`, filter theo label/threshold và join với transaction gốc.
- Feature attribution: đọc `feature_attribution.csv`, heatmap z theo từng feature cho mỗi transaction test set, kèm mean z theo feature và phân bố top_feature; filter theo label/loại fraud/transaction_id.

## Cách Chạy

Frontend cần được serve qua HTTP để browser đọc được CSV trong repo.

Chạy từ root repo:

```powershell
python -m http.server 5173
```

Mở:

```text
http://localhost:5173/frontend/
```

## Dữ Liệu Đang Đọc

```text
../data/transaction/transaction.csv
../exp/transaction/semi_supervised/split1/split0_hard/scores/transaction_scores_with_labels.csv
../exp/transaction/semi_supervised/split1/split0_hard/scores/feature_attribution.csv
```

## Ghi Chú

Đây là frontend tĩnh. Browser không thể tự chạy Python train/evaluate nếu không có backend local. Các nút `Run Training` và `Run Evaluate` đã đặt sẵn theo hướng gọi API:

```text
POST /api/train
POST /api/evaluate
```

Khi chưa có backend, UI vẫn tạo và copy command đầy đủ để chạy trong PowerShell.
