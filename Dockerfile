FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    coinor-cbc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app/packages:/app/apps:/app/apps/web
EXPOSE 8501 8000

CMD ["streamlit", "run", "apps/web/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
