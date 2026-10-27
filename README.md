# Phát hiện gian lận thẻ tín dụng – Nhóm 8 (Streamlit)

## 1. Huấn luyện (chạy 1 lần trên máy của bạn)
```bash
pip install -r requirements.txt
python train.py --data D:/fraudTrain.csv
```
Lệnh này tái hiện Giai đoạn 1–12 của notebook và tạo thư mục `artifacts/`
(`pipeline.joblib`, `metrics.json`, `eda.json`, `demo_test_sample.csv`; tổng vài MB).

## 2. Chạy thử trên máy
```bash
streamlit run app.py
```

## 3. Đưa lên Streamlit Community Cloud (nhận link công khai)
1. Tạo repo GitHub, đẩy **toàn bộ** thư mục này lên, **bao gồm `artifacts/`** (không đẩy `fraudTrain.csv`).
2. Vào https://share.streamlit.io → đăng nhập GitHub → **Create app**.
3. Chọn repo, nhánh `main`, Main file path: `app.py` → **Deploy**.
4. Sau ~2–3 phút bạn nhận link dạng `https://<tên-app>.streamlit.app`.
