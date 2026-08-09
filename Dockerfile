FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/home/app \
    GRADIO_ANALYTICS_ENABLED=False

WORKDIR /srv

RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0

COPY app/requirements.txt ./requirements.txt
RUN grep -viE '^\s*(torch)\b' requirements.txt > other-requirements.txt \
    && pip install -r other-requirements.txt

COPY app/ ./app/
COPY src/ ./src/
COPY dataset/ ./dataset/
COPY outputs/ ./outputs/

RUN useradd --create-home --uid 1000 app && chown -R app:app /srv /home/app
USER app

ENV PORT=8080
EXPOSE 8080

CMD ["python", "app/app.py"]
