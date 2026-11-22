# One container: builds the React dashboard, then serves it + the Flask API.
FROM node:20-alpine AS web
WORKDIR /web
COPY frontend/package.json ./
RUN npm install --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim
WORKDIR /app
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt
COPY backend/ backend/
COPY validation/simulation/ validation/simulation/
COPY validation/results/ validation/results/
COPY --from=web /web/dist frontend/dist
WORKDIR /app/backend
ENV PORT=10000 FRONTEND_DIST=/app/frontend/dist OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
EXPOSE 10000
# One worker (jobs live in memory), several threads for polling requests.
CMD gunicorn -b 0.0.0.0:${PORT} -w 1 --threads 8 --timeout 120 app:app
