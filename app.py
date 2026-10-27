from pathlib import Path
import shutil

source = Path("app_fraud_explain.py")
target = Path("app.py")

if not source.exists():
    raise FileNotFoundError(
        "Hãy đặt app_fraud_explain.py cùng thư mục với notebook."
    )

shutil.copyfile(source, target)

# Kiểm tra cú pháp trước khi sử dụng
compile(target.read_text(encoding="utf-8"), str(target), "exec")

print("Đã tạo app.py hoàn chỉnh và kiểm tra cú pháp thành công!")
