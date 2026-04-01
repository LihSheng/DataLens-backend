FROM python:3.11-slim

WORKDIR /app

# System dependencies for OCR and PDF parsing
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/data/uploads /app/data/chroma

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
