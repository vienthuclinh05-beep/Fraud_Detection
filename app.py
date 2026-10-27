from pathlib import Path

src_path = Path("/mnt/data/Văn bản đã dán (1)(3).txt")
src = src_path.read_text(encoding="utf-8")

# Add helper functions after classify_risk.
needle = '''def classify_risk(probability):
    """
    Phân loại giao dịch dựa trên xác suất gian lận.
    """

    if probability < LOW_THRESHOLD:
        return "Low risk", "Chấp nhận giao dịch"

    elif probability < HIGH_THRESHOLD:
        return "Review", "Cần kiểm tra thêm"

    return "High risk", "Đề xuất chặn giao dịch"
'''
addition = needle + '''


# ============================================================
# 5.1. GIẢI THÍCH DỰ ĐOÁN CHO MỘT GIAO DỊCH
# ============================================================

FEATURE_LABELS_VI = {
    "time_diff": "Thời gian từ giao dịch trước (giây)",
    "transactions_last_5min": "Số giao dịch trong 5 phút trước",
    "transactions_last_1h": "Số giao dịch trong 1 giờ trước",
    "transactions_last_24h": "Số giao dịch trong 24 giờ trước",
    "amount_sum_1h": "Tổng tiền giao dịch trong 1 giờ trước",
    "amount_sum_24h": "Tổng tiền giao dịch trong 24 giờ trước",
    "amt": "Số tiền giao dịch hiện tại",
    "customer_avg_amount": "Số tiền trung bình lịch sử của khách hàng",
    "customer_amt_zscore": "Mức khác biệt số tiền so với lịch sử khách hàng",
    "amt_zscore_cat": "Mức khác biệt số tiền trong cùng danh mục",
    "amt_ratio_cat_max": "Tỷ lệ số tiền so với mức cao nhất trước đó trong danh mục",
    "has_customer_history": "Có lịch sử giao dịch trước đó",
    "amt_jump_ratio": "Tỷ lệ số tiền so với giao dịch liền trước",
    "is_small_txn": "Giao dịch có số tiền nhỏ",
    "n_small_online_24h": "Số giao dịch nhỏ trực tuyến trong 24 giờ trước",
    "category": "Danh mục giao dịch",
    "is_online_cat": "Giao dịch thuộc danh mục trực tuyến",
    "is_new_merchant": "Cửa hàng mới với khách hàng",
    "hour_sin": "Đặc trưng thời điểm giao dịch (sin)",
    "hour_cos": "Đặc trưng thời điểm giao dịch (cos)",
    "is_night": "Giao dịch diễn ra ban đêm",
    "day_of_week": "Ngày trong tuần",
    "distance_zscore": "Mức khác biệt khoảng cách so với lịch sử",
    "if_score": "Điểm bất thường từ Isolation Forest",
}


def build_aligned_features(raw_df):
    """
    Tạo đặc trưng và khôi phục đúng thứ tự dòng của file đầu vào.
    Hàm tạo đặc trưng lịch sử có sắp xếp theo thời gian, vì vậy cần
    khôi phục thứ tự ban đầu trước khi ghép với kết quả dự đoán.
    """
    work = raw_df.copy().reset_index(drop=True)
    work["_original_row_position"] = range(len(work))
    features = build_feature_frame(work)
    features = (
        features.sort_values("_original_row_position", kind="stable")
        .drop(columns=["_original_row_position"])
        .reset_index(drop=True)
    )
    return features


def explain_one_transaction(raw_df, row_position, pipeline, top_n=8):
    """
    Giải thích dự đoán của XGBoost bằng feature contributions (SHAP values).
    Giá trị đóng góp dương đẩy điểm dự đoán về phía gian lận; giá trị âm
    kéo điểm dự đoán ra xa phía gian lận. Đây không phải xác suất riêng
    của từng đặc trưng.
    """
    import xgboost as xgb

    features = build_aligned_features(raw_df)
    X_processed = pipeline.preprocess(features)
    if_scores = pipeline._if_score(pipeline.if_model, X_processed)
    X_e2 = pipeline.build_e2(X_processed, if_scores)

    booster = pipeline.xgb.get_booster()
    contributions = booster.predict(
        xgb.DMatrix(X_e2[row_position:row_position + 1]),
        pred_contribs=True
    )[0]

    feature_names = getattr(pipeline, "feature_names_e2", None)
    if not feature_names or len(feature_names) != len(contributions) - 1:
        feature_names = [f"Đặc trưng {i + 1}" for i in range(len(contributions) - 1)]

    contribution_df = pd.DataFrame({
        "feature": feature_names,
        "contribution": contributions[:-1],
    })
    contribution_df["abs_contribution"] = contribution_df["contribution"].abs()
    contribution_df = contribution_df.sort_values(
        "abs_contribution", ascending=False
    )

    raw_features = features.iloc[row_position]
    rows = []
    for _, item in contribution_df.head(top_n).iterrows():
        feature = item["feature"]
        # Các biến one-hot có dạng category_xxx; giá trị của chúng nằm
        # trong ma trận đã biến đổi nên hiển thị tên danh mục ở tên biến.
        if feature.startswith("category_"):
            label = f"Danh mục: {feature[len('category_'):]}"
            value = "Biến mã hóa danh mục"
        else:
            label = FEATURE_LABELS_VI.get(feature, feature)
            value = raw_features.get(feature, "Đặc trưng đã biến đổi")
            if isinstance(value, (float, np.floating)):
                value = round(float(value), 4)
            elif isinstance(value, (np.integer,)):
                value = int(value)

        rows.append({
            "Yếu tố": label,
            "Giá trị giao dịch": str(value),
            "Đóng góp vào dự đoán": float(item["contribution"]),
            "Chiều tác động": (
                "Tăng điểm gian lận"
                if item["contribution"] > 0
                else "Giảm điểm gian lận"
                if item["contribution"] < 0
                else "Gần như không tác động"
            ),
        })

    explanation = pd.DataFrame(rows)
    return explanation, float(contributions[-1])
'''

if needle not in src:
    raise RuntimeError("Không tìm thấy vị trí chèn classify_risk")

src = src.replace(needle, addition, 1)

# Ensure the scoring features align with the original uploaded row order.
src = src.replace(
    '    features = build_feature_frame(raw.copy())',
    '    features = build_aligned_features(raw.copy())',
    1
)

# Save raw data for interactive explanations.
src = src.replace(
    '            st.session_state["batch_result"] = results\n'
    '            st.session_state["batch_filename"] = uploaded_file.name',
    '            st.session_state["batch_result"] = results\n'
    '            st.session_state["batch_raw"] = raw_df.copy().reset_index(drop=True)\n'
    '            st.session_state["batch_filename"] = uploaded_file.name',
    1
)

# Replace static dataframe with selectable dataframe.
old_table = '''    st.dataframe(
        display_results,
        use_container_width=True,
        hide_index=True,
        height=500,
    )
'''
new_table = '''    display_results = display_results.reset_index(drop=True)

    st.caption(
        "Chọn một dòng trong bảng để xem các đặc trưng đóng góp nhiều nhất "
        "vào dự đoán của mô hình."
    )
    table_event = st.dataframe(
        display_results,
        use_container_width=True,
        hide_index=True,
        height=500,
        on_select="rerun",
        selection_mode="single-row",
        key="fraud_transaction_table",
    )

    # --------------------------------------------------------
    # 12.3. GIẢI THÍCH GIAO DỊCH ĐƯỢC CHỌN
    # --------------------------------------------------------

    selected_rows = table_event.selection.rows if table_event is not None else []

    if selected_rows:
        selected_display_position = selected_rows[0]
        if 0 <= selected_display_position < len(filtered_results):
            selected_result = filtered_results.iloc[selected_display_position]
            transaction_number = int(selected_result["transaction_row"])
            raw_for_explanation = st.session_state.get("batch_raw")

            st.markdown("### Vì sao giao dịch này bị cảnh báo?")
            detail_col1, detail_col2, detail_col3 = st.columns(3)

            with detail_col1:
                st.metric("Giao dịch số", f"#{transaction_number}")

            with detail_col2:
                st.metric(
                    "Xác suất gian lận",
                    f"{float(selected_result['fraud_probability']):.2%}"
                )

            with detail_col3:
                st.metric("Mức rủi ro", str(selected_result["risk_level"]))

            st.write(
                "**Cách đọc:** đóng góp dương làm tăng điểm dự đoán gian lận; "
                "đóng góp âm làm giảm điểm dự đoán gian lận. Đây là đóng góp "
                "trên thang điểm của mô hình, không phải phần trăm xác suất."
            )

            if raw_for_explanation is None:
                st.warning(
                    "Không tìm thấy dữ liệu gốc trong phiên hiện tại. "
                    "Vui lòng tải file và chạy phân tích lại."
                )
            else:
                try:
                    explanation_df, _ = explain_one_transaction(
                        raw_for_explanation,
                        transaction_number - 1,
                        pipe,
                        top_n=8,
                    )

                    positive = explanation_df[
                        explanation_df["Đóng góp vào dự đoán"] > 0
                    ].copy()
                    negative = explanation_df[
                        explanation_df["Đóng góp vào dự đoán"] < 0
                    ].copy()

                    reason_col1, reason_col2 = st.columns(2)

                    with reason_col1:
                        st.markdown("#### Yếu tố làm tăng điểm gian lận")
                        if positive.empty:
                            st.info("Không có yếu tố đóng góp dương trong nhóm đặc trưng hiển thị.")
                        else:
                            st.dataframe(
                                positive.sort_values(
                                    "Đóng góp vào dự đoán", ascending=False
                                ),
                                use_container_width=True,
                                hide_index=True,
                            )

                    with reason_col2:
                        st.markdown("#### Yếu tố làm giảm điểm gian lận")
                        if negative.empty:
                            st.info("Không có yếu tố đóng góp âm trong nhóm đặc trưng hiển thị.")
                        else:
                            st.dataframe(
                                negative.sort_values(
                                    "Đóng góp vào dự đoán", ascending=True
                                ),
                                use_container_width=True,
                                hide_index=True,
                            )

                    st.caption(
                        "Giải thích được tính từ đóng góp của các đặc trưng trong "
                        "XGBoost cho giao dịch đã chọn. Đây là tín hiệu hỗ trợ "
                        "rà soát, không phải bằng chứng xác nhận gian lận."
                    )
                except Exception as e:
                    st.error(
                        "Chưa thể tạo giải thích cho giao dịch này. "
                        f"Chi tiết lỗi: {e}"
                    )
'''
if old_table not in src:
    raise RuntimeError("Không tìm thấy bảng kết quả để thay thế")
src = src.replace(old_table, new_table, 1)

# Renumber the following subsection headings to remain tidy.
src = src.replace(
    '# 12.3. TẢI KẾT QUẢ',
    '# 12.4. TẢI KẾT QUẢ',
    1
).replace(
    '# 12.4. ĐÁNH GIÁ NẾU FILE CÓ NHÃN THẬT',
    '# 12.5. ĐÁNH GIÁ NẾU FILE CÓ NHÃN THẬT',
    1
)

out_path = Path("/mnt/data/app_fraud_explain.py")
out_path.write_text(src, encoding="utf-8")
print(f"Đã tạo file: {out_path}")
print(f"Số dòng: {len(src.splitlines())}")
