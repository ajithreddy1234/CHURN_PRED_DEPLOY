# ---------- Base Image ----------
FROM python:3.10-slim

# ---------- Set Working Directory ----------
WORKDIR /app

# ---------- Install Dependencies ----------
# Copy only requirements first (for better caching)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# ---------- Copy Application Files ----------
COPY . .

# ---------- Set Environment Variables ----------
# Render dynamically sets $PORT, so we just expose a placeholder
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# ---------- Expose Port ----------
EXPOSE 8000

# ---------- Start Gunicorn Server ----------
# Gunicorn serves the Flask app: application:application
#   (file: application.py | Flask instance: application)
CMD ["sh", "-c", "gunicorn -w 2 -k gthread -b 0.0.0.0:${PORT} application:application"]
