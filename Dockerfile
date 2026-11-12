# Hugging Face Docker Space. The process listens on 7860 and does not run as root.
# https://huggingface.co/docs/hub/spaces-sdks-docker

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR $HOME/app

COPY --chown=user requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

COPY --chown=user . $HOME/app
RUN pip install --no-cache-dir --no-deps .

EXPOSE 7860

CMD ["uvicorn", "visualization_3d.app:app", "--host", "0.0.0.0", "--port", "7860"]
