#!/bin/bash
apt-get update && apt-get install -y wget unzip curl gnupg jq fonts-liberation libappindicator3-1 xdg-utils libasound2 libatk-bridge2.0-0 libatk1.0-0 libcups2 libdbus-1-3 libgdk-pixbuf2.0-0 libnspr4 libnss3 libx11-xcb1 libxcomposite1 libxdamage1 libxrandr2 libgbm1 libxshmfence1 libpango-1.0-0 libpangocairo-1.0-0 libatspi2.0-0 ca-certificates && \
CHROME_DIR="/home/site/wwwroot/bin/chrome" && \
CHROMEDRIVER_PATH="/home/site/wwwroot/bin/chromedriver" && \
if [ ! -f "$CHROME_DIR/chrome" ] || [ ! -f "$CHROMEDRIVER_PATH" ]; then \
    rm -rf /home/site/wwwroot/bin/chrome* /home/site/wwwroot/bin/chromedriver* && \
    VERSION=$(curl -sSL https://googlechromelabs.github.io/chrome-for-testing/last-known-good-versions-with-downloads.json | jq -r '.channels.Stable.version') && \
    wget -q "https://storage.googleapis.com/chrome-for-testing-public/${VERSION}/linux64/chrome-linux64.zip" && \
    wget -q "https://storage.googleapis.com/chrome-for-testing-public/${VERSION}/linux64/chromedriver-linux64.zip" && \
    unzip -q chrome-linux64.zip && unzip -q chromedriver-linux64.zip && \
    mkdir -p "$CHROME_DIR" && mv chrome-linux64/* "$CHROME_DIR/" && \
    mv chromedriver-linux64/chromedriver "$CHROMEDRIVER_PATH" && \
    chmod +x "$CHROMEDRIVER_PATH" && \
    ln -sf "$CHROME_DIR/chrome" /home/site/wwwroot/bin/chromium; \
fi && \
gunicorn -w 1 -k uvicorn.workers.UvicornWorker main:app --timeout 1790
