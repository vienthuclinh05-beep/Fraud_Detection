"""Tiền xử lý + Isolation Forest + XGBoost (E2) – gói gọn để lưu/tải bằng joblib."""
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from features import BEHAVIOR_COLS, CAT_COLS, FEATURE_COLS, NUMERIC_COLS

# Ma trận chi phí kinh tế (Giai đoạn 8 của notebook)
L_FN, C_REVIEW, C_MODEL = 100.0, 3.0, 0.05
COST_TP, COST_FN = C_REVIEW + C_MODEL, L_FN + C_MODEL
COST_FP, COST_TN = C_REVIEW + C_MODEL, C_MODEL
FINAL_THRESHOLD = 0.12
LOW_THRESHOLD = 0.05


def total_cost(tp, fp, fn, tn):
    return tp * COST_TP + fp * COST_FP + fn * COST_FN + tn * COST_TN


def decide(p, low=LOW_THRESHOLD, high=FINAL_THRESHOLD):
    p = np.asarray(p)
    return np.where(p < low, "Phê duyệt tự động (Approve)",
           np.where(p < high, "Xem xét thủ công (Manual Review)", "Chặn / Từ chối (Block)"))


class FraudPipeline:
    def __init__(self):
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = RobustScaler()
        self.ohe = OneHotEncoder(handle_unknown="ignore")
        self.if_model = None
        self.if_model_b = None
        self.xgb = None
        self.other_numeric = [c for c in NUMERIC_COLS if c != "customer_avg_amount"]

    # ---------- tiền xử lý (Giai đoạn 5) ----------
    def _prep_raw(self, feat: pd.DataFrame) -> pd.DataFrame:
        X = feat[FEATURE_COLS].copy()
        for c in NUMERIC_COLS:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        X[NUMERIC_COLS] = X[NUMERIC_COLS].replace([np.inf, -np.inf], np.nan)
        X["category"] = X["category"].fillna("__MISSING__").astype(str)
        X["customer_avg_amount"] = X["customer_avg_amount"].fillna(0)
        return X

    def fit_preprocess(self, feat):
        X = self._prep_raw(feat)
        X[self.other_numeric] = self.imputer.fit_transform(X[self.other_numeric])
        num = self.scaler.fit_transform(X[NUMERIC_COLS])
        ohe = self.ohe.fit_transform(X[CAT_COLS])
        self.ohe_names = self.ohe.get_feature_names_out(CAT_COLS).tolist()
        self._set_positions()
        return sparse.hstack([sparse.csr_matrix(num), ohe], format="csr")

    def preprocess(self, feat):
        X = self._prep_raw(feat)
        X[self.other_numeric] = self.imputer.transform(X[self.other_numeric])
        num = self.scaler.transform(X[NUMERIC_COLS])
        ohe = self.ohe.transform(X[CAT_COLS])
        return sparse.hstack([sparse.csr_matrix(num), ohe], format="csr")

    def _set_positions(self):
        total = len(NUMERIC_COLS) + len(self.ohe_names)
        self.behavior_pos = [NUMERIC_COLS.index(c) for c in BEHAVIOR_COLS]
        base_num = [i for i, c in enumerate(NUMERIC_COLS) if c not in BEHAVIOR_COLS]
        self.base_pos = base_num + list(range(len(NUMERIC_COLS), total))
        self.feature_names_e2 = (
            [NUMERIC_COLS[i] for i in base_num] + self.ohe_names
            + BEHAVIOR_COLS + ["if_score"]
        )

    # ---------- Isolation Forest (Giai đoạn 7) ----------
    @staticmethod
    def _if_score(model, X, batch=50_000):
        return np.concatenate([-model.decision_function(X[s:s + batch])
                               for s in range(0, X.shape[0], batch)])

    @staticmethod
    def _new_if():
        return IsolationForest(n_estimators=100, contamination="auto", random_state=42, n_jobs=-1)

    def fit_isolation_forest(self, X_train, y_train):
        normal = np.flatnonzero(np.asarray(y_train) == 0)
        n_fit = min(100_000, len(normal) // 2)
        rng = np.random.default_rng(42)
        sh = rng.permutation(normal)
        pos_a, pos_b = sh[:n_fit], sh[n_fit:2 * n_fit]
        self.if_model = self._new_if().fit(X_train[pos_a])
        self.if_model_b = self._new_if().fit(X_train[pos_b])
        train_score = self._if_score(self.if_model, X_train)
        train_score[pos_a] = self._if_score(self.if_model_b, X_train[pos_a])  # tránh rò rỉ
        return train_score

    # ---------- E2 (Giai đoạn 8) ----------
    def build_e2(self, X_processed, if_score):
        base = X_processed[:, self.base_pos]
        beh = X_processed[:, self.behavior_pos]
        return sparse.hstack([base, beh, sparse.csr_matrix(np.asarray(if_score).reshape(-1, 1))],
                             format="csr")

    # ---------- suy luận ----------
    def score_frame(self, feat: pd.DataFrame):
        Xp = self.preprocess(feat)
        s = self._if_score(self.if_model, Xp)
        prob = self.xgb.predict_proba(self.build_e2(Xp, s))[:, 1]
        return prob, s
