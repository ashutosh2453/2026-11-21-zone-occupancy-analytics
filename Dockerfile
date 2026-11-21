FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TORCH_HOME=/models RESULTS_DIR=/data
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md LICENSE ./
COPY occupancy ./occupancy
RUN pip install --no-cache-dir . \
 && python -c "from occupancy.detector import TorchvisionPersonDetector; TorchvisionPersonDetector()" \
 && useradd --create-home app && mkdir -p /data && chown -R app /data /models
USER app
EXPOSE 8000
CMD ["uvicorn", "occupancy.api:app", "--host", "0.0.0.0", "--port", "8000"]
