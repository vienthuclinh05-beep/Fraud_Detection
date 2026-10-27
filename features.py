"""Feature engineering – bê nguyên logic từ Giai đoạn 2 & 4 của notebook."""
import numpy as np
import pandas as pd

FEATURE_COLS = [
    "time_diff", "transactions_last_5min", "transactions_last_1h",
    "transactions_last_24h", "amount_sum_1h", "amount_sum_24h",
    "amt", "customer_avg_amount", "customer_amt_zscore", "amt_zscore_cat",
    "amt_ratio_cat_max", "has_customer_history",
    "amt_jump_ratio", "is_small_txn", "n_small_online_24h",
    "category", "is_online_cat", "is_new_merchant",
    "hour_sin", "hour_cos", "is_night", "day_of_week",
    "distance_zscore",
]
CAT_COLS = ["category"]
NUMERIC_COLS = [c for c in FEATURE_COLS if c not in CAT_COLS]
BEHAVIOR_COLS = [
    "time_diff", "transactions_last_5min", "transactions_last_1h",
    "transactions_last_24h", "amount_sum_1h", "amount_sum_24h",
    "customer_avg_amount", "customer_amt_zscore", "amt_ratio_cat_max",
    "amt_jump_ratio", "n_small_online_24h", "has_customer_history",
    "distance_zscore",
]
ONLINE_CATEGORIES = {"shopping_net", "misc_net"}
RAW_REQUIRED = [
    "trans_date_trans_time", "cc_num", "amt", "category", "merchant",
    "lat", "long", "merch_lat", "merch_long",
]
NS_MIN = 60 * 1_000_000_000
WINDOWS_NS = {"5min": 5 * NS_MIN, "1h": 60 * NS_MIN, "24h": 24 * 60 * NS_MIN}


def haversine(lat1, lon1, lat2, lon2):
    r = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return r * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def add_stateless_features(df: pd.DataFrame) -> pd.DataFrame:
    """Đặc trưng không phụ thuộc lịch sử (số tiền, thời gian, danh mục)."""
    df = df.copy()
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")
    if "distance_km" not in df.columns:
        df["distance_km"] = haversine(df["lat"], df["long"], df["merch_lat"], df["merch_long"])
    df["amt_log"] = np.log1p(df["amt"])
    df["hour"] = df["trans_date_trans_time"].dt.hour
    df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 6)).astype("int8")
    df["is_online_cat"] = df["category"].isin(ONLINE_CATEGORIES).astype("int8")
    df["is_small_txn"] = (df["amt"] < 20).astype("int8")
    return df


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Đặc trưng lịch sử theo từng thẻ (chỉ dùng giao dịch TRƯỚC thời điểm hiện tại)."""
    df = df.sort_values("trans_date_trans_time", kind="stable").reset_index(drop=True)
    n = len(df)
    time_ns = df["trans_date_trans_time"].to_numpy(dtype="datetime64[ns]").astype("int64")

    out = {c: np.zeros(n) for c in [
        "customer_amt_zscore", "time_diff", "transactions_last_5min",
        "transactions_last_1h", "transactions_last_24h", "amount_sum_1h",
        "amount_sum_24h", "amt_zscore_cat", "amt_ratio_cat_max",
        "amt_jump_ratio", "n_small_online_24h", "is_new_merchant",
        "distance_zscore"]}
    out["customer_avg_amount"] = np.full(n, np.nan)
    out["has_customer_history"] = np.zeros(n, dtype=np.int8)

    amt_all = df["amt"].to_numpy(float)
    dist_all = df["distance_km"].to_numpy(float)
    cat_all = df["category"].to_numpy()
    mer_all = df["merchant"].astype(object).to_numpy()
    online_all = df["is_online_cat"].to_numpy(np.int8)
    small_online_all = ((df["is_small_txn"] == 1) & (df["is_online_cat"] == 1)).to_numpy(np.int8)

    for _, g in df.groupby("cc_num", sort=False, dropna=False):
        pos = g.index.to_numpy()
        times = time_ns[pos]
        amounts = amt_all[pos]
        distances = dist_all[pos]
        categories = cat_all[pos]
        merchants = mer_all[pos]
        small_online = small_online_all[pos]
        m = len(pos)

        prev_end = np.searchsorted(times, times, side="left")
        has_hist = prev_end > 0
        out["has_customer_history"][pos] = has_hist.astype(np.int8)

        # --- thống kê số tiền theo lịch sử khách hàng
        a_pre = np.concatenate(([0.0], np.cumsum(amounts)))
        a_sq = np.concatenate(([0.0], np.cumsum(amounts ** 2)))
        mean = np.full(m, np.nan)
        mean[has_hist] = a_pre[prev_end[has_hist]] / prev_end[has_hist]
        out["customer_avg_amount"][pos] = mean
        var = np.zeros(m)
        var[has_hist] = a_sq[prev_end[has_hist]] / prev_end[has_hist] - mean[has_hist] ** 2
        std = np.sqrt(np.maximum(var, 0.0))
        z = np.zeros(m)
        ok = has_hist & (std > 0)
        z[ok] = (amounts[ok] - mean[ok]) / std[ok]
        out["customer_amt_zscore"][pos] = z

        # --- khoảng cách thời gian
        td = np.zeros(m)
        td[has_hist] = (times[has_hist] - times[prev_end[has_hist] - 1]) / 1e9
        out["time_diff"][pos] = td

        # --- cửa sổ thời gian
        for suffix, w in WINDOWS_NS.items():
            left = np.minimum(np.searchsorted(times, times - w, side="left"), prev_end)
            cnt = prev_end - left
            if suffix == "5min":
                out["transactions_last_5min"][pos] = cnt
            elif suffix == "1h":
                out["transactions_last_1h"][pos] = cnt
                out["amount_sum_1h"][pos] = a_pre[prev_end] - a_pre[left]
            else:
                out["transactions_last_24h"][pos] = cnt
                out["amount_sum_24h"][pos] = a_pre[prev_end] - a_pre[left]

        # --- tỷ lệ so với giao dịch liền trước
        prev_amt = np.zeros(m)
        prev_amt[has_hist] = amounts[prev_end[has_hist] - 1]
        jr = np.zeros(m)
        vp = has_hist & (prev_amt > 0)
        jr[vp] = amounts[vp] / prev_amt[vp]
        out["amt_jump_ratio"][pos] = jr

        # --- giao dịch nhỏ + online trong 24h
        so_pre = np.concatenate(([0], np.cumsum(small_online)))
        l24 = np.minimum(np.searchsorted(times, times - WINDOWS_NS["24h"], side="left"), prev_end)
        out["n_small_online_24h"][pos] = so_pre[prev_end] - so_pre[l24]

        # --- z-score khoảng cách
        d_pre = np.concatenate(([0.0], np.cumsum(distances)))
        d_sq = np.concatenate(([0.0], np.cumsum(distances ** 2)))
        dm = np.zeros(m)
        dm[has_hist] = d_pre[prev_end[has_hist]] / prev_end[has_hist]
        dv = np.zeros(m)
        dv[has_hist] = d_sq[prev_end[has_hist]] / prev_end[has_hist] - dm[has_hist] ** 2
        ds = np.sqrt(np.maximum(dv, 0.0))
        dz = np.zeros(m)
        okd = has_hist & (ds > 0)
        dz[okd] = (distances[okd] - dm[okd]) / ds[okd]
        out["distance_zscore"][pos] = dz

        # --- đặc trưng theo danh mục
        cat_z = np.zeros(m)
        cat_ratio = np.zeros(m)
        for category in pd.unique(categories):
            idx = np.flatnonzero(categories == category)
            ct, ca = times[idx], amounts[idx]
            cpe = np.searchsorted(ct, ct, side="left")
            chh = cpe > 0
            pre = np.concatenate(([0.0], np.cumsum(ca)))
            sq = np.concatenate(([0.0], np.cumsum(ca ** 2)))
            cm = np.zeros(len(idx))
            cv = np.zeros(len(idx))
            cm[chh] = pre[cpe[chh]] / cpe[chh]
            cv[chh] = sq[cpe[chh]] / cpe[chh] - cm[chh] ** 2
            cs = np.sqrt(np.maximum(cv, 0.0))
            lz = np.zeros(len(idx))
            okc = chh & (cs > 0)
            lz[okc] = (ca[okc] - cm[okc]) / cs[okc]
            lmax = np.zeros(len(idx))
            pmax = np.maximum.accumulate(ca)
            lmax[chh] = pmax[cpe[chh] - 1]
            lr = np.zeros(len(idx))
            vm = lmax > 0
            lr[vm] = ca[vm] / lmax[vm]
            cat_z[idx] = lz
            cat_ratio[idx] = lr
        out["amt_zscore_cat"][pos] = cat_z
        out["amt_ratio_cat_max"][pos] = cat_ratio

        # --- merchant mới (giao dịch cùng timestamp không tính vào lịch sử của nhau)
        new_m = np.zeros(m, dtype=np.int8)
        seen = set()
        i = 0
        while i < m:
            j = i + 1
            while j < m and times[j] == times[i]:
                j += 1
            for k in range(i, j):
                new_m[k] = 0 if pd.isna(merchants[k]) else int(merchants[k] not in seen)
            for k in range(i, j):
                if not pd.isna(merchants[k]):
                    seen.add(merchants[k])
            i = j
        out["is_new_merchant"][pos] = new_m

    for k, v in out.items():
        df[k] = v
    return df


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Từ dữ liệu giao dịch thô -> bảng có đủ 23 biến đầu vào của mô hình."""
    return add_history_features(add_stateless_features(df))
