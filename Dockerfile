
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

WORKDIR /app

# Thư viện hệ thống
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Cài dependencies của project
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Cài spaces cho decorator @spaces.GPU
RUN pip install --no-cache-dir spaces

# Sao chép mã nguồn
COPY app.py .
COPY src/ ./src/
COPY assets/ ./assets/

EXPOSE 7860

CMD ["python", "app.py"]
