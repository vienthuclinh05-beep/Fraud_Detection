"""Huấn luyện lại toàn bộ pipeline của notebook và xuất file vào thư mục artifacts/.

Cách chạy:
    python train.py --data path/to/fraudTrain.csv
"""
import argparse, json, os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from xgboost import XGBClassifier

from features import FEATURE_COLS, build_feature_frame
from model_pipeline import (FINAL_THRESHOLD, LOW_THRESHOLD, FraudPipeline,
                            decide, total_cost)

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="Đường dẫn tới fraudTrain.csv")
ap.add_argument("--out", default="artifacts")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

# ---- Giai đoạn 1-2: tải + làm sạch ----
print("Đọc dữ liệu...")
df = pd.read_csv(args.data).drop(columns=["Unnamed: 0"], errors="ignore")
df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")
df = df.sort_values("trans_date_trans_time", kind="stable").reset_index(drop=True)

# ---- EDA tóm tắt cho trang web ----
eda = {
    "n_rows": int(len(df)), "n_fraud": int(df["is_fraud"].sum()),
    "fraud_rate_pct": float(df["is_fraud"].mean() * 100),
    "amt_mean": float(df["amt"].mean()), "amt_median": float(df["amt"].median()),
    "amt_mean_fraud": float(df.loc[df.is_fraud == 1, "amt"].mean()),
    "amt_mean_normal": float(df.loc[df.is_fraud == 0, "amt"].mean()),
    "date_min": str(df["trans_date_trans_time"].min()),
    "date_max": str(df["trans_date_trans_time"].max()),
    "fraud_by_category": (df.groupby("category")["is_fraud"].agg(["sum", "count", "mean"])
                          .rename(columns={"sum": "fraud", "count": "total", "mean": "rate"})
                          .reset_index().to_dict("records")),
    "fraud_by_hour": (df.assign(hour=df["trans_date_trans_time"].dt.hour)
                      .groupby("hour")["is_fraud"].agg(["sum", "count", "mean"])
                      .rename(columns={"sum": "fraud", "count": "total", "mean": "rate"})
                      .reset_index().to_dict("records")),
    "categories": sorted(df["category"].dropna().unique().tolist()),
}

# ---- Giai đoạn 3-4: chia theo thời gian 70/15/15 + feature engineering ----
n = len(df)
tr_end, va_end = int(n * 0.70), int(n * 0.85)
df["_split"] = "train"
df.loc[tr_end:va_end - 1, "_split"] = "validation"
df.loc[va_end:, "_split"] = "test"
print("Tạo đặc trưng (có thể mất vài phút)...")
feat = build_feature_frame(df)
tr = feat[feat["_split"] == "train"]
va = feat[feat["_split"] == "validation"]
te = feat[feat["_split"] == "test"]
y_tr, y_va, y_te = (x["is_fraud"].to_numpy(int) for x in (tr, va, te))

# ---- Giai đoạn 5: tiền xử lý ----
pipe = FraudPipeline()
Xp_tr = pipe.fit_preprocess(tr)
Xp_va, Xp_te = pipe.preprocess(va), pipe.preprocess(te)
spw = float((y_tr == 0).sum() / (y_tr == 1).sum())

# ---- Giai đoạn 7: Isolation Forest ----
print("Huấn luyện Isolation Forest...")
s_tr = pipe.fit_isolation_forest(Xp_tr, y_tr)
s_va = pipe._if_score(pipe.if_model, Xp_va)
s_te = pipe._if_score(pipe.if_model, Xp_te)

# ---- Giai đoạn 8-9: XGBoost + E2 ----
print("Huấn luyện XGBoost...")
E2_tr, E2_va, E2_te = (pipe.build_e2(a, b) for a, b in
                       [(Xp_tr, s_tr), (Xp_va, s_va), (Xp_te, s_te)])
pipe.xgb = XGBClassifier(
    n_estimators=300, learning_rate=0.1, max_depth=6, subsample=0.8,
    colsample_bytree=0.8, scale_pos_weight=spw, tree_method="hist",
    eval_metric="aucpr", n_jobs=-1, random_state=42,
).fit(E2_tr, y_tr)

# ---- Giai đoạn 9: quét threshold trên Validation ----
p_va = pipe.xgb.predict_proba(E2_va)[:, 1]
grid = np.round(np.arange(0.01, 0.99, 0.01), 2)
def cost_at(y, p, t):
    tn, fp, fn, tp = confusion_matrix(y, (p >= t).astype(int), labels=[0, 1]).ravel()
    return total_cost(tp, fp, fn, tn)
val_costs = [cost_at(y_va, p_va, t) for t in grid]
best_val_t = float(grid[int(np.argmin(val_costs))])
print(f"Threshold tối ưu chi phí trên Validation: {best_val_t:.2f} "
      f"(notebook chốt {FINAL_THRESHOLD})")

# ---- Giai đoạn 10: đánh giá Test ----
p_te = pipe.xgb.predict_proba(E2_te)[:, 1]
def evaluate(t):
    pred = (p_te >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()
    return {"threshold": t, "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "precision": float(precision_score(y_te, pred, zero_division=0)),
            "recall": float(recall_score(y_te, pred)),
            "f1": float(f1_score(y_te, pred)),
            "cost": float(total_cost(tp, fp, fn, tn))}
metrics = {
    "roc_auc": float(roc_auc_score(y_te, p_te)),
    "pr_auc": float(average_precision_score(y_te, p_te)),
    "at_default": evaluate(0.50), "at_final": evaluate(FINAL_THRESHOLD),
    "best_val_threshold": best_val_t, "final_threshold": FINAL_THRESHOLD,
    "low_threshold": LOW_THRESHOLD, "scale_pos_weight": spw,
    "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
    "val_cost_curve": {"threshold": grid.tolist(), "cost": [float(c) for c in val_costs]},
}
u, c = np.unique(decide(p_te), return_counts=True)
metrics["decision_counts"] = {k: int(v) for k, v in zip(u, c)}

# ---- Feature importance (gain) ----
booster = pipe.xgb.get_booster()
gain = booster.get_score(importance_type="gain")
names = pipe.feature_names_e2
metrics["importance"] = sorted(
    [{"feature": nm, "gain": float(gain.get(f"f{i}", 0.0))} for i, nm in enumerate(names)],
    key=lambda d: -d["gain"])

# ---- Lưu artifacts ----
joblib.dump(pipe, f"{args.out}/pipeline.joblib", compress=3)
with open(f"{args.out}/eda.json", "w", encoding="utf-8") as f:
    json.dump(eda, f, ensure_ascii=False)
with open(f"{args.out}/metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False)

# Mẫu Test nhỏ để demo trên web (toàn bộ fraud, tối đa 600 + 3.000 giao dịch hợp lệ)
rng = np.random.default_rng(42)
fr = np.flatnonzero(y_te == 1)
fr = rng.choice(fr, size=min(600, len(fr)), replace=False)
no = rng.choice(np.flatnonzero(y_te == 0), size=3000, replace=False)
idx = np.sort(np.concatenate([fr, no]))
demo = te.iloc[idx][["trans_date_trans_time", "cc_num", "merchant", "category",
                     "amt", "distance_km", "is_fraud"]].copy()
demo["probability"] = p_te[idx]
demo["if_score"] = s_te[idx]
demo.to_csv(f"{args.out}/demo_test_sample.csv", index=False)
print("Xong! File đã lưu trong", args.out)
