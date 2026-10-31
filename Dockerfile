# Sử dụng base image Python 3.11 slim tối ưu dung lượng
FROM python:3.11-slim

# Thiết lập biến môi trường
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Cài đặt các thư viện hệ thống cần thiết cho PyTorch, OpenCV, curl (healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    libglib2.0-0 \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt dependencies Python trước để tận dụng Docker layer cache
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir spaces

# Tạo sẵn cấu trúc thư mục cần thiết
RUN mkdir -p /app/data/processed_v1 /app/checkpoints /app/assets

# Sao chép mã nguồn ứng dụng và trọng số mô hình
COPY app.py .
COPY src/ ./src/
COPY assets/ ./assets/
COPY checkpoints/ ./checkpoints/

# Tạo người dùng non-root để đảm bảo tiêu chuẩn an toàn bảo mật
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

# Healthcheck định kỳ kiểm tra ứng dụng phản hồi HTTP 200
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

CMD ["python", "app.py"]
