# LiteAgent — multi-stage Docker build
FROM python:3.11-slim AS builder

WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir .[api]

# Runtime stage
FROM python:3.11-slim
WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY . .

# Create non-root user for security
RUN groupadd -r liteagent && useradd -r -g liteagent -d /app -s /sbin/nologin liteagent \
    && mkdir -p /data /config /home/liteagent/.liteagent \
    && chown -R liteagent:liteagent /app /data /config /home/liteagent
USER liteagent
ENV HOME=/home/liteagent

# Data and config mount points
VOLUME ["/data", "/config"]
ENV LITEAGENT_CONFIG=/config/config.json
ENV LITEAGENT_DB=/data/memory.db

EXPOSE 8080

CMD ["python", "-m", "liteagent", "--channel", "api", "-c", "/config/config.json"]
