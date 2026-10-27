...
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from features import FEATURE_COLS, RAW_REQUIRED, build_feature_frame
from model_pipeline import (
    C_MODEL,
    C_REVIEW,
    COST_FN,
    COST_FP,
    COST_TN,
    COST_TP,
    L_FN,
    FraudPipeline,
)


# ============================================================
# 1. CẤU HÌNH ỨNG DỤNG
# ============================================================

st.set_page_config(
    page_title="Đánh giá rủi ro giao dịch thẻ",
    page_icon="💳",
    layout="wide",
)

LOW_THRESHOLD = 0.05
HIGH_THRESHOLD = 0.12
MAX_ROWS = 1_000_000


# ============================================================
# 2. PHÂN LOẠI MỨC ĐỘ RỦI RO
# ============================================================

def tier_of(probability: float) -> int:
    if probability < LOW_THRESHOLD:
        return 1
    if probability < HIGH_THRESHOLD:
        return 2
    return 3


TIER_LABELS = {
    1: "Rủi ro thấp",
    2: "Cần xác minh",
    3: "Rủi ro cao",
}

TIER_ACTIONS = {
    1: "Đề xuất phê duyệt tự động",
    2: "Đề xuất kiểm tra thủ công",
    3: "Ưu tiên kiểm tra / xem xét từ chối",
}


# ============================================================
# 3. TẢI MÔ HÌNH ĐÃ HUẤN LUYỆN
# ============================================================

MODEL_PATHS = (
    os.path.join("artifacts", "pipeline.joblib"),
    "pipeline.joblib",
)

MODEL_PATH = next(
    (path for path in MODEL_PATHS if os.path.isfile(path)),
    None,
)

if MODEL_PATH is None:
    st.error(
        "Không tìm thấy pipeline.joblib. "
        "Hãy đặt file trong thư mục artifacts/ "
        "hoặc cùng thư mục với app_risk_giao_dich.py."
    )
    st.stop()


@st.cache_resource
def load_pipeline(path: str):
    return joblib.load(path)


try:
    pipe = load_pipeline(MODEL_PATH)
except Exception as error:
    st.error("Không thể tải mô hình đã huấn luyện.")
    with st.expander("Chi tiết lỗi kỹ thuật"):
        st.code(str(error))
    st.stop()


# ============================================================
# 4. ĐỌC FILE CSV / EXCEL
# ============================================================

def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    """Đọc file CSV hoặc Excel được tải lên."""

    name = uploaded_file.name.lower()
    uploaded_file.seek(0)

    if name.endswith(".csv"):
        try:
            return pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            return pd.read_csv(
                uploaded_file,
                encoding="latin-1",
            )

    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)

    raise ValueError(
        "Định dạng không được hỗ trợ. "
        "Hãy tải lên file CSV hoặc Excel."
    )
# ============================================================
# 5. XỬ LÝ TOÀN BỘ GIAO DỊCH
# ============================================================

def process_batch(raw_df: pd.DataFrame, pipeline):
    """
    Tạo đặc trưng và chấm điểm toàn bộ giao dịch.
    Giữ nguyên thứ tự các dòng để có thể giải thích từng giao dịch.
    """

    if raw_df.empty:
        raise ValueError("File tải lên không có giao dịch.")

    if len(raw_df) > MAX_ROWS:
        raise ValueError(
            f"File có {len(raw_df):,} dòng, vượt giới hạn "
            f"{MAX_ROWS:,} dòng của ứng dụng."
        )

    missing_columns = [
        column
        for column in RAW_REQUIRED
        if column not in raw_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "File thiếu các cột bắt buộc: "
            + ", ".join(missing_columns)
        )

    raw = raw_df.copy().reset_index(drop=True)

    # Tạo đặc trưng theo đúng pipeline của dự án.
    features = build_feature_frame(raw.copy())

    if len(features) != len(raw):
        raise ValueError(
            "Số dòng sau khi tạo đặc trưng không khớp "
            "với dữ liệu đầu vào."
        )

    missing_features = [
        column
        for column in FEATURE_COLS
        if column not in features.columns
    ]

    if missing_features:
        raise ValueError(
            "Thiếu đặc trưng mà mô hình yêu cầu: "
            + ", ".join(missing_features)
        )

    # Đảm bảo thứ tự đặc trưng đúng như lúc huấn luyện.
    model_features = (
        features[FEATURE_COLS]
        .copy()
        .reset_index(drop=True)
    )

    probabilities, anomaly_scores = pipeline.score_frame(
        model_features
    )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    ).reshape(-1)

    if len(probabilities) != len(raw):
        raise ValueError(
            "Số xác suất dự đoán không khớp số giao dịch."
        )

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "Mô hình trả về xác suất không hợp lệ."
        )

    # Giới hạn giá trị do xác suất phải nằm trong [0, 1].
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError(
            "Xác suất mô hình trả về nằm ngoài khoảng [0, 1]."
        )

    # Bảng kết quả giữ lại thông tin giao dịch cần thiết.
    preferred_columns = [
        "trans_date_trans_time",
        "cc_num",
        "merchant",
        "category",
        "amt",
    ]

    display_columns = [
        column
        for column in preferred_columns
        if column in raw.columns
    ]

    results = raw[display_columns].copy().reset_index(drop=True)

    # Giữ nhãn thật nếu có, chỉ để đối chiếu sau dự đoán.
    if "is_fraud" in raw.columns:
results["is_fraud"] = raw["is_fraud"].to_numpy()

    # Mã dòng gốc để chọn giao dịch và đối chiếu kết quả.
    results.insert(
        0,
        "transaction_row",
        np.arange(1, len(raw) + 1),
    )

    results["probability"] = probabilities
    results["probability_pct"] = (
        probabilities * 100
    ).round(2)

    # Lưu điểm bất thường nếu pipeline trả về.
    if anomaly_scores is not None:
        anomaly_scores = np.asarray(
            anomaly_scores,
            dtype=float,
        ).reshape(-1)

        if len(anomaly_scores) == len(raw):
            results["if_score"] = anomaly_scores
        else:
            results["if_score"] = np.nan
    else:
        results["if_score"] = np.nan

    results["tier"] = [
        tier_of(float(probability))
        for probability in probabilities
    ]

    results["risk_level"] = results["tier"].map(
        TIER_LABELS
    )

    results["recommended_action"] = results["tier"].map(
        TIER_ACTIONS
    )

    return results, model_features


# ============================================================
# 6. GIAO DIỆN TIÊU ĐỀ
# ============================================================

st.title("Đánh giá rủi ro giao dịch thẻ")

st.write(
    "Tải lên file giao dịch để hệ thống tự động tạo đặc trưng, "
    "đánh giá xác suất gian lận và phân loại mức độ rủi ro. "
    "Bạn có thể chọn một giao dịch bất kỳ để xem giải thích SHAP."
)

m1, m2, m3 = st.columns(3)

m1.metric("Trạng thái mô hình", "Đã tải")
m2.metric("Ngưỡng cần xác minh", f"≥ {LOW_THRESHOLD:.0%}")
m3.metric("Ngưỡng rủi ro cao", f"≥ {HIGH_THRESHOLD:.0%}")


# ============================================================
# 7. THANH BÊN
# ============================================================

with st.sidebar:
    st.header("Thông tin hệ thống")

    st.success("Đã tải mô hình thành công.")
    st.caption(f"Mô hình: `{MODEL_PATH}`")

    st.divider()

    st.subheader("Ngưỡng phân loại")

    st.write(
        f"**Rủi ro thấp:** xác suất < {LOW_THRESHOLD:.0%}"
    )

    st.write(
        f"**Cần xác minh:** {LOW_THRESHOLD:.0%} đến "
        f"dưới {HIGH_THRESHOLD:.0%}"
    )

    st.write(
        f"**Rủi ro cao:** xác suất ≥ {HIGH_THRESHOLD:.0%}"
    )

    st.divider()

    st.caption(
        "Kết quả là đánh giá của mô hình, không phải kết luận "
        "chắc chắn rằng giao dịch gian lận."
    )


with st.expander("Giải thích quy tắc phân loại"):
    st.markdown(
        f"""
- **Rủi ro thấp:** xác suất gian lận dưới {LOW_THRESHOLD:.0%}.
- **Cần xác minh:** xác suất từ {LOW_THRESHOLD:.0%} đến dưới {HIGH_THRESHOLD:.0%}.
- **Rủi ro cao:** xác suất từ {HIGH_THRESHOLD:.0%} trở lên.

Xác suất là đầu ra của mô hình, không phải bằng chứng chắc chắn về gian lận.
Kết quả phụ thuộc vào chất lượng dữ liệu và mức độ đầy đủ của lịch sử giao dịch.
"""
    )


# ============================================================
# 8. UPLOAD FILE GIAO DỊCH
# ============================================================

st.divider()
st.subheader("1. Tải file giao dịch")

st.write(
    "Hỗ trợ CSV, XLSX và XLS. Mỗi dòng tương ứng với một giao dịch."
)

with st.expander("Các cột dữ liệu bắt buộc"):
    st.code(
        ", ".join(str(column) for column in RAW_REQUIRED),
        language="text",
    )

    st.caption(
        "Tên cột phải tương thích với dữ liệu dùng để xây dựng "
        "đặc trưng và huấn luyện mô hình."
    )

uploaded_file = st.file_uploader(
    "Chọn file dữ liệu giao dịch",
    type=["csv", "xlsx", "xls"],
    help=(
        "Tải lên file giao dịch. Với các đặc trưng hành vi, "
        "nên cung cấp lịch sử giao dịch đầy đủ theo thời gian."
    ),
)


# ============================================================
# 9. ĐỌC FILE VÀ CHẤM ĐIỂM
# ============================================================

if uploaded_file is not None:

    # Nếu người dùng tải file mới, xóa kết quả của file cũ
    # để tránh hiển thị nhầm dữ liệu giữa hai file.
    if st.session_state.get("current_upload_name") != uploaded_file.name:
        st.session_state.pop("batch_result", None)
        st.session_state.pop("batch_features", None)
        st.session_state.pop("last_shap_row_id", None)
        st.session_state.pop("last_shap_table", None)
        st.session_state["current_upload_name"] = uploaded_file.name

    try:
        raw = read_uploaded_file(uploaded_file)
        raw.columns = raw.columns.astype(str).str.strip()

        if raw.empty:
            st.error("File không có giao dịch.")
            st.stop()

        missing_columns = [
            column
            for column in RAW_REQUIRED
            if column not in raw.columns
        ]

        if missing_columns:
            st.error(
                "File thiếu các cột bắt buộc: "
                + ", ".join(missing_columns)
            )
            st.write("Các cột hiện có:")
            st.code(
                ", ".join(str(column) for column in raw.columns),
                language="text",
            )
            st.stop()

        if len(raw) > MAX_ROWS:
            st.error(
                f"File có {len(raw):,} dòng, vượt giới hạn "
                f"{MAX_ROWS:,} dòng."
            )
            st.stop()

        c1, c2, c3 = st.columns(3)

        c1.metric("Số giao dịch", f"{len(raw):,}")
        c2.metric("Số cột", f"{len(raw.columns):,}")
        c3.metric(
            "Số giá trị thiếu",
            f"{int(raw.isna().sum().sum()):,}",
        )

        with st.expander("Xem trước dữ liệu đầu vào"):
            st.dataframe(
raw.head(10),
                use_container_width=True,
                hide_index=True,
            )

        if st.button(
            "Phân tích toàn bộ giao dịch",
            type="primary",
            use_container_width=True,
        ):
            with st.spinner(
                "Đang tạo đặc trưng và chấm điểm giao dịch..."
            ):
                results, model_features = process_batch(
                    raw,
                    pipe,
                )

            st.session_state["batch_result"] = results
            st.session_state["batch_features"] = model_features
            st.session_state["batch_filename"] = uploaded_file.name

            # Xóa giải thích SHAP cũ khi phân tích lại.
            st.session_state.pop("last_shap_row_id", None)
            st.session_state.pop("last_shap_table", None)

            st.success(
                f"Đã phân tích thành công {len(results):,} giao dịch."
            )

    except Exception as error:
        st.error("Không thể xử lý file.")
        with st.expander("Chi tiết lỗi kỹ thuật"):
            st.code(str(error))


# ============================================================
# 10. HIỂN THỊ KẾT QUẢ CHẤM ĐIỂM
# ============================================================

result = st.session_state.get("batch_result")

if result is not None:

    st.divider()
    st.subheader("2. Kết quả chấm điểm toàn bộ giao dịch")

    st.caption(
        f"File: {st.session_state.get('batch_filename', '—')} | "
        f"Tổng số giao dịch: {len(result):,}"
    )

    counts = result["risk_level"].value_counts()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Tổng giao dịch", f"{len(result):,}")
    c2.metric(
        "Rủi ro thấp",
        f"{int(counts.get('Rủi ro thấp', 0)):,}",
    )
    c3.metric(
        "Cần xác minh",
        f"{int(counts.get('Cần xác minh', 0)):,}",
    )
    c4.metric(
        "Rủi ro cao",
        f"{int(counts.get('Rủi ro cao', 0)):,}",
    )

    f1, f2 = st.columns([1, 2])

    with f1:
        selected_risks = st.multiselect(
            "Lọc mức rủi ro",
            options=[
                "Rủi ro thấp",
                "Cần xác minh",
                "Rủi ro cao",
            ],
            default=[
                "Rủi ro thấp",
                "Cần xác minh",
                "Rủi ro cao",
            ],
        )

    with f2:
        sort_option = st.selectbox(
            "Sắp xếp kết quả",
            options=[
                "Xác suất gian lận giảm dần",
                "Theo thứ tự file gốc",
                "Xác suất gian lận tăng dần",
            ],
        )

    filtered = result[
        result["risk_level"].isin(selected_risks)
    ].copy()

    if sort_option == "Xác suất gian lận giảm dần":
        filtered = filtered.sort_values(
"probability",
            ascending=False,
        )

    elif sort_option == "Xác suất gian lận tăng dần":
        filtered = filtered.sort_values(
            "probability",
            ascending=True,
        )

    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True,
        height=500,
        column_config={
            "transaction_row": st.column_config.NumberColumn(
                "Dòng giao dịch"
            ),
            "probability": st.column_config.NumberColumn(
                "Xác suất (0–1)",
                format="%.4f",
            ),
            "probability_pct": st.column_config.NumberColumn(
                "Xác suất gian lận (%)",
                format="%.2f%%",
            ),
            "if_score": st.column_config.NumberColumn(
                "Điểm bất thường",
                format="%.4f",
            ),
            "tier": st.column_config.NumberColumn(
                "Mức rủi ro (mã)"
            ),
        },
    )

    st.caption(
        f"Đang hiển thị {len(filtered):,}/{len(result):,} giao dịch."
    )

    # --------------------------------------------------------
    # 10.1. TẢI KẾT QUẢ
    # --------------------------------------------------------

    csv_bytes = result.to_csv(
        index=False,
        encoding="utf-8-sig",
    )

    st.download_button(
        "Tải xuống toàn bộ kết quả (CSV)",
        data=csv_bytes,
        file_name="ket_qua_danh_gia_rui_ro.csv",
        mime="text/csv",
        type="primary",
        use_container_width=True,
    )

    # --------------------------------------------------------
    # 10.2. ĐỐI CHIẾU NHÃN THỰC TẾ
    # --------------------------------------------------------

    with st.expander(
        "Đối chiếu với nhãn thực tế (nếu có cột is_fraud)",
        expanded=False,
    ):

        if "is_fraud" not in result.columns:
            st.info(
                "File không có cột is_fraud nên không thể tính "
                "các chỉ số đối chiếu với nhãn thực tế."
            )

        else:
            y = pd.to_numeric(
                result["is_fraud"],
                errors="coerce",
            )

            valid = y.isin([0, 1])

            if not valid.any():
                st.warning(
                    "Cột is_fraud không có nhãn 0/1 hợp lệ."
                )

            else:
                # Dùng ngưỡng rủi ro cao để đánh giá nhị phân.
                pred_high = (
                    result.loc[valid, "probability"].to_numpy()
                    >= HIGH_THRESHOLD
                )

                y_true = y.loc[valid].astype(int).to_numpy()

                tp = int(
                    ((pred_high == 1) & (y_true == 1)).sum()
                )
                fp = int(
((pred_high == 1) & (y_true == 0)).sum()
                )
                fn = int(
                    ((pred_high == 0) & (y_true == 1)).sum()
                )
                tn = int(
                    ((pred_high == 0) & (y_true == 0)).sum()
                )

                a, b, c, d = st.columns(4)

                a.metric("TP — phát hiện đúng", f"{tp:,}")
                b.metric("FP — cảnh báo nhầm", f"{fp:,}")
                c.metric("FN — bỏ sót", f"{fn:,}")
                d.metric("TN — nhận diện đúng", f"{tn:,}")

                st.caption(
                    f"Đối chiếu nhị phân dùng ngưỡng {HIGH_THRESHOLD:.0%}. "
                    "Đây là thống kê trên file hiện tại, không thay thế "
                    "đánh giá mô hình trên tập kiểm thử độc lập."
                )


    # ========================================================
    # 11. GIẢI THÍCH GIAO DỊCH ĐƯỢC CHỌN BẰNG SHAP
    # ========================================================

    st.divider()
    st.subheader("3. Giải thích một giao dịch bằng SHAP")

    st.write(
        "Chọn số thứ tự giao dịch để xem các đặc trưng đóng góp "
        "nhiều nhất vào xác suất gian lận do pipeline hiện tại dự đoán."
    )

    result_with_id = result.copy()

    selected_row_id = st.selectbox(
        "Chọn giao dịch cần giải thích",
        options=result_with_id["transaction_row"].astype(int).tolist(),
        format_func=lambda row_id: (
            f"Dòng {row_id} | "
            f"P(gian lận) = "
            f"{float(result_with_id.loc[result_with_id['transaction_row'] == row_id, 'probability'].iloc[0]):.2%}"
            f" | "
            f"{result_with_id.loc[result_with_id['transaction_row'] == row_id, 'risk_level'].iloc[0]}"
        ),
        key="shap_selected_transaction",
    )

    selected_idx = int(selected_row_id) - 1

    selected_result = result_with_id.iloc[selected_idx]

    selected_features = st.session_state.get("batch_features")

    if (
        selected_features is None
        or len(selected_features) != len(result_with_id)
    ):
        st.warning(
            "Không tìm thấy đặc trưng tương ứng. "
            "Hãy tải file và phân tích lại."
        )

    else:
        selected_feature_row = (
            selected_features.iloc[[selected_idx]]
            [FEATURE_COLS]
            .copy()
        )

        detail_cols = [
            column
            for column in [
                "trans_date_trans_time",
                "cc_num",
                "merchant",
                "category",
                "amt",
            ]
            if column in result_with_id.columns
        ]

        detail = (
            selected_result[detail_cols]
            .to_frame("Giá trị")
            .reset_index()
.rename(columns={"index": "Thông tin"})
        )

        st.markdown("### Thông tin giao dịch được chọn")

        st.dataframe(
            detail,
            use_container_width=True,
            hide_index=True,
        )

        c1, c2 = st.columns(2)

        c1.metric(
            "Xác suất gian lận",
            f"{float(selected_result['probability']):.2%}",
        )

        c2.metric(
            "Mức rủi ro",
            selected_result["risk_level"],
        )

        st.write(
            f"**Hành động đề xuất:** "
            f"{selected_result['recommended_action']}"
        )

        if st.button(
            "Tạo giải thích SHAP cho giao dịch này",
            type="primary",
            key="run_shap",
        ):

            try:
                import shap

                feature_names = list(FEATURE_COLS)

                # Lấy mẫu nền từ file đang phân tích.
                # Với file rất lớn, chỉ sử dụng tối đa 20 dòng làm nền.
                background_n = min(20, len(selected_features))

                background = selected_features.sample(
                    n=background_n,
                    random_state=42,
                ).copy()

                def predict_probability(x):
                    """
                    SHAP gọi hàm này để đánh giá đầu ra của pipeline.
                    Không tạo hoặc huấn luyện mô hình dự đoán mới.
                    """
                    x_df = pd.DataFrame(
                        x,
                        columns=feature_names,
                    )

                    p_values, _ = pipe.score_frame(x_df)

                    return np.asarray(
                        p_values,
                        dtype=float,
                    ).reshape(-1)

                explainer = shap.Explainer(
                    predict_probability,
                    background,
                    algorithm="permutation",
                )

                max_evals = max(
                    2 * len(feature_names) + 1,
                    100,
                )

                with st.spinner(
                    "Đang tính SHAP cho giao dịch được chọn..."
                ):
                    shap_exp = explainer(
                        selected_feature_row,
                        max_evals=max_evals,
                    )

                shap_values = np.asarray(
                    shap_exp.values
                ).reshape(-1)

                shap_table = pd.DataFrame({
                    "Đặc trưng": feature_names,
                    "Giá trị đặc trưng": [
                        selected_feature_row.iloc[0][column]
                        for column in feature_names
                    ],
                    "SHAP value": shap_values,
"Độ lớn đóng góp": np.abs(shap_values),
                })

                shap_table = shap_table.sort_values(
                    "Độ lớn đóng góp",
                    ascending=False,
                )

                st.markdown("### Các đặc trưng tác động mạnh nhất")

                st.caption(
                    "SHAP value dương cho biết đặc trưng làm tăng "
                    "đầu ra xác suất gian lận; SHAP value âm cho biết "
                    "đặc trưng làm giảm đầu ra này. Đây là đóng góp "
                    "tương đối theo mẫu nền, không phải quan hệ nhân quả."
                )

                st.dataframe(
                    shap_table.head(15),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "SHAP value": st.column_config.NumberColumn(
                            format="%.6f"
                        ),
                        "Độ lớn đóng góp": st.column_config.NumberColumn(
                            format="%.6f"
                        ),
                    },
                )

                # Diễn giải các đặc trưng làm tăng xác suất.
                st.markdown("### Diễn giải kết quả")

                positive = (
                    shap_table[shap_table["SHAP value"] > 0]
                    .head(5)
                )

                negative = (
                    shap_table[shap_table["SHAP value"] < 0]
                    .sort_values("SHAP value")
                    .head(5)
                )

                if not positive.empty:
                    st.markdown(
                        "**Các đặc trưng đóng góp làm tăng xác suất gian lận:**"
                    )

                    for _, row in positive.iterrows():
                        st.write(
                            f"- `{row['Đặc trưng']}`: "
                            f"giá trị = `{row['Giá trị đặc trưng']}`, "
                            f"SHAP = `{row['SHAP value']:+.6f}`"
                        )

                if not negative.empty:
                    st.markdown(
                        "**Các đặc trưng đóng góp làm giảm xác suất gian lận:**"
                    )

                    for _, row in negative.iterrows():
                        st.write(
                            f"- `{row['Đặc trưng']}`: "
                            f"giá trị = `{row['Giá trị đặc trưng']}`, "
                            f"SHAP = `{row['SHAP value']:+.6f}`"
                        )

                st.session_state["last_shap_row_id"] = selected_row_id
                st.session_state["last_shap_table"] = shap_table

            except ImportError:
                st.error(
"Chưa cài thư viện SHAP. "
                    "Hãy thêm shap vào requirements.txt "
                    "hoặc cài bằng lệnh: pip install shap"
                )

            except Exception as error:
                st.error(
                    "Không tạo được giải thích SHAP cho giao dịch này."
                )

                with st.expander("Chi tiết lỗi SHAP"):
                    st.code(str(error))

        # Hiển thị lại bảng SHAP sau mỗi lần Streamlit rerun.
        if (
            st.session_state.get("last_shap_row_id") == selected_row_id
            and "last_shap_table" in st.session_state
        ):
            st.markdown("### Bảng SHAP gần nhất")

            st.dataframe(
                st.session_state["last_shap_table"].head(15),
                use_container_width=True,
                hide_index=True,
            )


    # ========================================================
    # 12. MÔ HÌNH CHI PHÍ
    # ========================================================

    st.divider()
    st.subheader("4. Mô hình chi phí giả định")

    st.caption(
        "Các tham số được lấy từ model_pipeline.py. "
        "Đây là giả định phục vụ nghiên cứu, không phải chi phí "
        "thực tế đã được xác nhận của ngân hàng."
    )

    cost_info1, cost_info2, cost_info3 = st.columns(3)

    cost_info1.metric(
        "Tổn thất bỏ sót gian lận",
        f"${L_FN:,.2f}",
    )

    cost_info2.metric(
        "Chi phí xác minh",
        f"${C_REVIEW:,.2f}",
    )

    cost_info3.metric(
        "Chi phí chấm điểm",
        f"${C_MODEL:,.2f}",
    )

    cost_matrix = pd.DataFrame(
        [
            [COST_TP, COST_FN],
            [COST_FP, COST_TN],
        ],
        index=[
            "Thực tế: Fraud",
            "Thực tế: Normal",
        ],
        columns=[
            "Dự đoán: Cảnh báo",
            "Dự đoán: Không cảnh báo",
        ],
    )

    st.markdown("### Ma trận chi phí")

    st.dataframe(
        cost_matrix.style.format("${:,.2f}"),
        use_container_width=True,
    )

    st.markdown("### Công thức chi phí")

    st.code(
        f"Total Cost = FN × {L_FN:g} "
        f"+ (TP + FP) × {C_REVIEW:g} "
        f"+ N × {C_MODEL:g}",
        language="text",
    )

    if "is_fraud" in result.columns:

        y = pd.to_numeric(
            result["is_fraud"],
            errors="coerce",
        )

        valid = y.isin([0, 1])

        if valid.any():

            actual = y.loc[valid].astype(int).to_numpy()

            # Theo mô hình chi phí 3 mức, mọi giao dịch từ ngưỡng
            # 5% trở lên đều được chuyển sang xác minh/cảnh báo.
            alert = (
                result.loc[valid, "probability"].to_numpy()
                >= LOW_THRESHOLD
)

            tp = int(
                ((actual == 1) & (alert == 1)).sum()
            )

            fp = int(
                ((actual == 0) & (alert == 1)).sum()
            )

            fn = int(
                ((actual == 1) & (alert == 0)).sum()
            )

            tn = int(
                ((actual == 0) & (alert == 0)).sum()
            )

            total_n = len(actual)

            total_cost_value = (
                fn * L_FN
                + (tp + fp) * C_REVIEW
                + total_n * C_MODEL
            )

            c1, c2, c3, c4, c5 = st.columns(5)

            c1.metric("TP — cảnh báo đúng", f"{tp:,}")
            c2.metric("FP — cảnh báo nhầm", f"{fp:,}")
            c3.metric("FN — bỏ sót", f"{fn:,}")
            c4.metric("TN — không cảnh báo đúng", f"{tn:,}")
            c5.metric(
                "Tổng chi phí ước tính",
                f"${total_cost_value:,.2f}",
            )

            st.caption(
                f"Đánh giá trên {total_n:,} dòng có nhãn 0/1 hợp lệ. "
                f"Giao dịch có xác suất từ {LOW_THRESHOLD:.0%} trở lên "
                "được tính là cần xác minh; giao dịch gian lận dưới "
                "ngưỡng này được tính là bỏ sót."
            )

        else:
            st.info(
                "Cột is_fraud không có nhãn 0/1 hợp lệ, "
                "nên không thể tính chi phí dựa trên nhãn thực tế."
            )

    else:
        # Không có nhãn thật: chỉ tính chi phí kỳ vọng theo xác suất.
        probabilities = result["probability"].astype(float).to_numpy()

        alert = probabilities >= LOW_THRESHOLD

        # Chi phí xác minh cho các giao dịch được cảnh báo.
        review_cost = int(alert.sum()) * C_REVIEW

        # Với giao dịch được thông qua, tính tổn thất kỳ vọng
        # bằng xác suất gian lận nhân với mức tổn thất giả định.
        missed_fraud_expected_cost = (
            probabilities[~alert].sum() * L_FN
        )

        # Chi phí chấm điểm cho toàn bộ giao dịch.
        scoring_cost = len(probabilities) * C_MODEL

        expected_cost = (
            review_cost
            + missed_fraud_expected_cost
            + scoring_cost
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Chi phí xác minh",
            f"${review_cost:,.2f}",
        )

        c2.metric(
            "Tổn thất kỳ vọng do bỏ sót",
            f"${missed_fraud_expected_cost:,.2f}",
        )

        c3.metric(
            "Chi phí chấm điểm",
            f"${scoring_cost:,.2f}",
        )

        st.metric(
            "Tổng chi phí kỳ vọng",
            f"${expected_cost:,.2f}",
        )

        st.caption(
"File không có nhãn thực tế nên không thể xác định TP, FP, "
            "FN và TN. Kết quả trên là chi phí kỳ vọng dựa trên xác suất "
            "mô hình, không phải chi phí thực tế."
        )

    with st.expander("Giải thích các thành phần chi phí"):

        st.markdown(
            f"""
- **FN:** giao dịch gian lận bị bỏ sót, tổn thất giả định **${L_FN:.2f}/giao dịch**.
- **TP + FP:** các giao dịch được chuyển sang xác minh, chi phí **${C_REVIEW:.2f}/giao dịch**.
- **N:** tổng số giao dịch được chấm điểm, chi phí **${C_MODEL:.2f}/giao dịch**.
- **Công thức:** `FN × L_FN + (TP + FP) × C_REVIEW + N × C_MODEL`.

Các mức chi phí được lấy từ cấu hình mô hình của dự án. Cần bảo đảm cách tính này thống nhất với mô hình chi phí trong notebook.
"""
        )

else:
    st.info(
        "Hãy tải lên file giao dịch và nhấn "
        "'Phân tích toàn bộ giao dịch' để xem kết quả."
    )


# ============================================================
# 13. CHÂN TRANG
# ============================================================

st.divider()

st.caption(
    "Fraud Detection System | Công cụ hỗ trợ sàng lọc giao dịch. "
    "Các cảnh báo cần được đánh giá cùng bối cảnh nghiệp vụ."
)
