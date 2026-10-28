FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GOAL_SETTINGS_RUNTIME_DIR=/app/runtime \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    LLM_ENABLED=false

WORKDIR /app

RUN addgroup --system --gid 10001 appgroup \
    && adduser --system --uid 10001 --ingroup appgroup --home /app appuser

COPY requirements.lock pyproject.toml README.md LICENSE ./
RUN python -m pip install --upgrade pip \
    && python -m pip install --requirement requirements.lock

COPY app ./app
COPY core ./core
COPY scripts ./scripts
COPY .streamlit ./.streamlit
RUN python -m pip install --no-deps . \
    && mkdir -p /app/runtime \
    && chown -R appuser:appgroup /app

USER appuser
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)" || exit 1

CMD ["streamlit", "run", "app/main.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
