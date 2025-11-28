FROM python:3.10-slim-bookworm

# Install system dependencies for Playwright (as root)
RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-liberation \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libatspi2.0-0 \
    libcairo2 \
    libcurl4 \
    libdbus-1-3 \
    libdrm2 \
    libgbm1 \
    libglib2.0-0 \
    libnspr4 \
    libnss3 \
    libpango-1.0-0 \
    libx11-6 \
    libxcb1 \
    libxcomposite1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxrandr2 \
    libxrender1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies and Playwright
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir playwright && \
    playwright install chromium

# Set working directory
WORKDIR /app

# Copy your code
COPY . .

# Install your app's Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port (Render sets $PORT)
EXPOSE 8000

# Start the app with Gunicorn (binds to $PORT)
CMD gunicorn -k uvicorn.workers.UvicornWorker app:app --bind 0.0.0.0:$PORT
