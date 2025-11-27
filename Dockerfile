FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

# 1. Install dependencies first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 2. Copy ALL files (app.py, agents.py, chroma_db/, frontend/) to the container
# This ensures your code and the database are present.
COPY . .

# 3. Optional: Download the ML model during build time
# This prevents the app from downloading 80MB+ on every startup (Cold Start).
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-mpnet-base-v2')"

# 4. Corrected Command
# Uses 'app:app' because your file is named 'app.py' and the FastAPI instance is 'app'
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]