import os
import joblib
import pandas as pd
import streamlit as st

from features import RAW_REQUIRED, build_feature_frame


# ============================================================
# 1. CẤU HÌNH ỨNG DỤNG
# ============================================================

LOW_THRESHOLD = 0.05
HIGH_THRESHOLD = 0.12
MAX_ROWS = 1_000_000

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. GIAO DIỆN
# ============================================================

st.title("Hệ thống phát hiện giao dịch gian lận")

st.markdown(
    "Tải lên dữ liệu giao dịch để hệ thống đánh giá xác suất gian lận, "
    "phân loại mức độ rủi ro và đưa ra hành động đề xuất."
)

# ============================================================
# 3. TẢI MÔ HÌNH
# ============================================================

@st.cache_resource
def load_pipeline():
    possible_paths = [
        os.path.join("artifacts", "pipeline.joblib"),
        "pipeline.joblib",
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return joblib.load(path), path

    return None, None


pipe, model_path = load_pipeline()


# ============================================================
# 4. ĐỌC FILE CSV / EXCEL
# ============================================================

def read_uploaded_file(uploaded_file):
    """
    Đọc file CSV hoặc Excel được tải lên.
    Trả về DataFrame chứa toàn bộ giao dịch.
    """

    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            df = pd.read_csv(
                uploaded_file,
                encoding="latin-1",
            )

    elif filename.endswith((".xlsx", ".xls")):
        uploaded_file.seek(0)
        df = pd.read_excel(uploaded_file)

    else:
        raise ValueError(
            "Định dạng file không được hỗ trợ. "
            "Vui lòng tải lên file CSV hoặc Excel."
        )

    # Chuẩn hóa tên cột để tránh lỗi do khoảng trắng
    df.columns = df.columns.astype(str).str.strip()

    return df


# ============================================================
# 5. PHÂN LOẠI MỨC ĐỘ RỦI RO
# ============================================================

def classify_risk(probability):
    """
    Phân loại giao dịch dựa trên xác suất gian lận.
    """

    if probability < LOW_THRESHOLD:
        return "Low risk", "Chấp nhận giao dịch"

    elif probability < HIGH_THRESHOLD:
        return "Review", "Cần kiểm tra thêm"

    return "High risk", "Đề xuất chặn giao dịch"


# ============================================================
# 6. XỬ LÝ TOÀN BỘ GIAO DỊCH TRONG FILE
# ============================================================

def process_batch(raw_df, pipeline):
    """
    Tiền xử lý và chấm điểm tất cả giao dịch trong file.
    Giữ nguyên thứ tự giao dịch đầu vào.
    """

    if raw_df.empty:
        raise ValueError("File tải lên không có dữ liệu.")

    if len(raw_df) > MAX_ROWS:
        raise ValueError(
            f"File có {len(raw_df):,} giao dịch, vượt giới hạn "
            f"{MAX_ROWS:,} giao dịch mỗi lần xử lý."
        )

    # Kiểm tra các cột bắt buộc
    missing_columns = [
        col for col in RAW_REQUIRED
        if col not in raw_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "File thiếu các cột bắt buộc: "
            + ", ".join(missing_columns)
        )

    # Giữ dữ liệu gốc để hiển thị kết quả
    raw = raw_df.copy().reset_index(drop=True)

    # Tạo đặc trưng theo đúng pipeline đã xây dựng
    features = build_feature_frame(raw.copy())

    if len(features) != len(raw):
        raise ValueError(
            "Số lượng giao dịch sau tiền xử lý không khớp "
            "với dữ liệu đầu vào."
        )

    # Chấm điểm bằng pipeline đã huấn luyện
    probabilities, anomaly_scores = pipeline.score_frame(features)

    probabilities = pd.Series(probabilities).reset_index(drop=True)

    if len(probabilities) != len(raw):
        raise ValueError(
            "Số xác suất dự đoán không khớp với số giao dịch đầu vào."
        )

    # Tạo bảng kết quả
    results = pd.DataFrame(index=range(len(raw)))

    # Giữ lại các thông tin giao dịch quan trọng nếu có
    display_columns = [
        "trans_date_trans_time",
        "cc_num",
        "merchant",
        "category",
        "amt",
    ]

    for col in display_columns:
        if col in raw.columns:
            results[col] = raw[col].values

    # Nếu file có nhãn thật, giữ lại để đánh giá mô hình
    if "is_fraud" in raw.columns:
        results["is_fraud"] = raw["is_fraud"].values

    # Xác suất gian lận
    results["fraud_probability"] = probabilities.values
    results["fraud_probability_pct"] = (
        probabilities.values * 100
    ).round(2)

    # Điểm bất thường nếu pipeline trả về
    if anomaly_scores is not None:
        anomaly_scores = pd.Series(
            anomaly_scores
        ).reset_index(drop=True)

        if len(anomaly_scores) == len(raw):
            results["anomaly_score"] = anomaly_scores.values

    # Phân loại rủi ro
    classifications = [
        classify_risk(float(probability))
        for probability in probabilities
    ]

    results["risk_level"] = [
        item[0] for item in classifications
    ]

    results["recommended_action"] = [
        item[1] for item in classifications
    ]

    # Thêm số thứ tự giao dịch để dễ đối chiếu
    results.insert(
        0,
        "transaction_row",
        range(1, len(results) + 1),
    )

    return results


# ============================================================
# 7. TIÊU ĐỀ ỨNG DỤNG
# ============================================================

st.markdown(
    '<div class="main-title">Hệ thống phát hiện giao dịch gian lận</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="sub-title">
    Tải lên dữ liệu giao dịch để hệ thống đánh giá xác suất gian lận,
    phân loại mức độ rủi ro và đưa ra hành động đề xuất.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 8. THANH BÊN
# ============================================================

with st.sidebar:
    st.header("Thông tin hệ thống")

    if pipe is not None:
        st.success("Đã tải mô hình thành công.")
        st.caption(f"Mô hình: `{model_path}`")
    else:
        st.error("Không tìm thấy file mô hình.")
        st.caption(
            "Vui lòng kiểm tra artifacts/pipeline.joblib "
            "hoặc pipeline.joblib."
        )

    st.divider()

    st.subheader("Ngưỡng phân loại")

    st.write(
        f"**Rủi ro thấp:** Xác suất < {LOW_THRESHOLD:.0%}"
    )

    st.write(
        f"**Cần xem xét:** {LOW_THRESHOLD:.0%} đến "
        f"dưới {HIGH_THRESHOLD:.0%}"
    )

    st.write(
        f"**Rủi ro cao:** Xác suất ≥ {HIGH_THRESHOLD:.0%}"
    )

    st.divider()

    st.caption(
        "Kết quả là điểm đánh giá của mô hình, "
        "không phải kết luận chắc chắn rằng giao dịch gian lận."
    )


# ============================================================
# 9. KIỂM TRA MÔ HÌNH
# ============================================================

if pipe is None:
    st.error(
        "Ứng dụng chưa thể chấm điểm vì không tải được mô hình. "
        "Hãy kiểm tra đường dẫn file pipeline.joblib."
    )
    st.stop()


# ============================================================
# 10. UPLOAD FILE
# ============================================================

st.subheader("1. Tải lên dữ liệu giao dịch")

st.write(
    "Chọn file CSV hoặc Excel chứa các giao dịch cần kiểm tra. "
    "Hệ thống sẽ xử lý toàn bộ giao dịch trong file."
)

uploaded_file = st.file_uploader(
    "Chọn file dữ liệu giao dịch",
    type=["csv", "xlsx", "xls"],
    accept_multiple_files=False,
    max_upload_size=1024,
    help="Hỗ trợ CSV, XLSX, XLS; dung lượng tối đa 1 GB.",
)

with st.expander("Xem các cột dữ liệu bắt buộc"):
    st.write(
        "File tải lên cần có đầy đủ các cột đầu vào mà "
        "pipeline yêu cầu:"
    )

    st.code(
        "\n".join(str(col) for col in RAW_REQUIRED),
        language="text",
    )

    st.caption(
        "Tên cột phải khớp với dữ liệu dùng trong bước "
        "tiền xử lý và huấn luyện mô hình."
    )


# ============================================================
# 11. CHẤM ĐIỂM DỮ LIỆU
# ============================================================

if uploaded_file is not None:

    st.divider()
    st.subheader("2. Kiểm tra dữ liệu đầu vào")

    try:
        raw_df = read_uploaded_file(uploaded_file)

        if raw_df.empty:
            st.error("File không có giao dịch để xử lý.")
            st.stop()

        missing_columns = [
            col for col in RAW_REQUIRED
            if col not in raw_df.columns
        ]

        if missing_columns:
            st.error(
                "File thiếu các cột bắt buộc: "
                + ", ".join(missing_columns)
            )

            st.write("Các cột hiện có trong file:")
            st.code(
                "\n".join(str(col) for col in raw_df.columns),
                language="text",
            )

            st.stop()

        if len(raw_df) > MAX_ROWS:
            st.error(
                f"File có {len(raw_df):,} giao dịch. "
                f"Giới hạn hiện tại là {MAX_ROWS:,} giao dịch."
            )
            st.stop()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Số giao dịch",
                f"{len(raw_df):,}",
            )

        with col2:
            st.metric(
                "Số cột dữ liệu",
                f"{len(raw_df.columns):,}",
            )

        with col3:
            st.metric(
                "Giá trị bị thiếu",
                f"{int(raw_df.isna().sum().sum()):,}",
            )

        with st.expander("Xem trước dữ liệu đầu vào"):
            st.dataframe(
                raw_df.head(10),
                use_container_width=True,
                hide_index=True,
            )

        if st.button(
            "Phân tích toàn bộ giao dịch",
            type="primary",
            use_container_width=True,
        ):

            with st.spinner(
                "Đang tiền xử lý dữ liệu và chấm điểm giao dịch..."
            ):
                results = process_batch(raw_df, pipe)

            # Lưu kết quả vào session
            st.session_state["batch_result"] = results
            st.session_state["batch_filename"] = uploaded_file.name

            st.success(
                f"Đã hoàn tất phân tích {len(results):,} giao dịch."
            )

    except Exception as e:
        st.error(f"Không thể xử lý file: {e}")


# ============================================================
# 12. HIỂN THỊ KẾT QUẢ
# ============================================================

if "batch_result" in st.session_state:

    results = st.session_state["batch_result"].copy()

    st.divider()
    st.subheader("3. Kết quả phân tích")

    st.caption(
        f"File kết quả: {st.session_state.get('batch_filename', '')}"
    )

    total_transactions = len(results)

    low_count = int(
        (results["risk_level"] == "Low risk").sum()
    )

    review_count = int(
        (results["risk_level"] == "Review").sum()
    )

    high_count = int(
        (results["risk_level"] == "High risk").sum()
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Tổng giao dịch",
            f"{total_transactions:,}",
        )

    with col2:
        st.metric(
            "Rủi ro thấp",
            f"{low_count:,}",
            f"{low_count / total_transactions:.1%}"
            if total_transactions else "0%",
        )

    with col3:
        st.metric(
            "Cần xem xét",
            f"{review_count:,}",
            f"{review_count / total_transactions:.1%}"
            if total_transactions else "0%",
        )

    with col4:
        st.metric(
            "Rủi ro cao",
            f"{high_count:,}",
            f"{high_count / total_transactions:.1%}"
            if total_transactions else "0%",
        )

    # --------------------------------------------------------
    # 12.1. BỘ LỌC KẾT QUẢ
    # --------------------------------------------------------

    st.markdown("### Lọc giao dịch")

    filter_col1, filter_col2 = st.columns([1, 2])

    with filter_col1:
        selected_risks = st.multiselect(
            "Mức độ rủi ro",
            options=[
                "Low risk",
                "Review",
                "High risk",
            ],
            default=[
                "Low risk",
                "Review",
                "High risk",
            ],
        )

    with filter_col2:
        search_text = st.text_input(
            "Tìm kiếm giao dịch",
            placeholder=(
                "Nhập mã thẻ, tên cửa hàng hoặc loại giao dịch..."
            ),
        )

    filtered_results = results[
        results["risk_level"].isin(selected_risks)
    ].copy()

    if search_text.strip():
        search_text = search_text.strip().lower()

        searchable_columns = [
            col for col in [
                "cc_num",
                "merchant",
                "category",
                "transaction_row",
            ]
            if col in filtered_results.columns
        ]

        if searchable_columns:
            mask = pd.Series(
                False,
                index=filtered_results.index,
            )

            for col in searchable_columns:
                mask |= (
                    filtered_results[col]
                    .astype(str)
                    .str.lower()
                    .str.contains(
                        search_text,
                        regex=False,
                        na=False,
                    )
                )

            filtered_results = filtered_results[mask]

    # Sắp xếp theo xác suất gian lận giảm dần
    filtered_results = filtered_results.sort_values(
        "fraud_probability",
        ascending=False,
    )

    st.write(
        f"Hiển thị **{len(filtered_results):,}** / "
        f"**{total_transactions:,}** giao dịch."
    )

    # --------------------------------------------------------
    # 12.2. HIỂN THỊ BẢNG
    # --------------------------------------------------------

    display_results = filtered_results.copy()

    if "fraud_probability" in display_results.columns:
        display_results["fraud_probability"] = (
            display_results["fraud_probability"].map(
                lambda x: f"{x:.4%}"
            )
        )

    if "fraud_probability_pct" in display_results.columns:
        display_results["fraud_probability_pct"] = (
            display_results["fraud_probability_pct"].map(
                lambda x: f"{x:.2f}%"
            )
        )

    st.dataframe(
        display_results,
        use_container_width=True,
        hide_index=True,
        height=500,
    )

    # --------------------------------------------------------
    # 12.3. TẢI KẾT QUẢ
    # --------------------------------------------------------

    st.markdown("### Tải kết quả về máy")

    download_col1, download_col2 = st.columns(2)

    with download_col1:
        csv_data = results.to_csv(
            index=False,
            encoding="utf-8-sig",
        )

        st.download_button(
            label="Tải toàn bộ kết quả (CSV)",
            data=csv_data,
            file_name="fraud_detection_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with download_col2:
        filtered_csv = filtered_results.to_csv(
            index=False,
            encoding="utf-8-sig",
        )

        st.download_button(
            label="Tải kết quả đang lọc (CSV)",
            data=filtered_csv,
            file_name="filtered_fraud_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # --------------------------------------------------------
    # 12.4. ĐÁNH GIÁ NẾU FILE CÓ NHÃN THẬT
    # --------------------------------------------------------

    if "is_fraud" in results.columns:

        st.divider()
        st.subheader("4. Đánh giá đối chiếu với nhãn thực tế")

        st.caption(
            "Phần này chỉ xuất hiện khi file đầu vào có cột "
            "`is_fraud`. Ngưỡng phân loại dùng để đối chiếu là "
            f"{HIGH_THRESHOLD:.0%}."
        )

        try:
            actual = pd.to_numeric(
                results["is_fraud"],
                errors="coerce",
            )

            valid_mask = actual.isin([0, 1])

            actual = actual[valid_mask].astype(int)

            predicted = (
                results.loc[valid_mask, "fraud_probability"]
                >= HIGH_THRESHOLD
            ).astype(int)

            tp = int(((actual == 1) & (predicted == 1)).sum())
            tn = int(((actual == 0) & (predicted == 0)).sum())
            fp = int(((actual == 0) & (predicted == 1)).sum())
            fn = int(((actual == 1) & (predicted == 0)).sum())

            metric1, metric2, metric3, metric4 = st.columns(4)

            with metric1:
                st.metric("True Positive", f"{tp:,}")

            with metric2:
                st.metric("True Negative", f"{tn:,}")

            with metric3:
                st.metric("False Positive", f"{fp:,}")

            with metric4:
                st.metric("False Negative", f"{fn:,}")

            if (tp + fn) > 0:
                recall = tp / (tp + fn)
                st.write(
                    f"**Recall:** {recall:.2%} "
                    "— tỷ lệ giao dịch gian lận thực tế "
                    "được mô hình phát hiện."
                )

            if (tp + fp) > 0:
                precision = tp / (tp + fp)
                st.write(
                    f"**Precision:** {precision:.2%} "
                    "— tỷ lệ giao dịch mô hình cảnh báo gian lận "
                    "thực sự là gian lận."
                )

            if not valid_mask.all():
                st.warning(
                    "Một số giá trị trong cột `is_fraud` không phải "
                    "0 hoặc 1 nên đã được loại khỏi phần đánh giá."
                )

        except Exception as e:
            st.warning(
                f"Không thể tính các chỉ số đối chiếu: {e}"
            )


# ============================================================
# 13. CHÂN TRANG
# ============================================================

st.divider()

st.caption(
    "Fraud Detection System | Công cụ hỗ trợ sàng lọc giao dịch. "
    "Các cảnh báo cần được đánh giá cùng bối cảnh nghiệp vụ."
)
