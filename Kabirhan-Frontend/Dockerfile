# ============================================================
# Race Vision — Frontend (React → Nginx)
# Multi-stage: Node build + Nginx serve
# ============================================================

# ── Stage 1: Build ───────────────────────────────────────────
FROM node:22-alpine AS build

WORKDIR /app

# Install dependencies first (cache layer)
COPY package.json package-lock.json* yarn.lock* pnpm-lock.yaml* ./
RUN if [ -f pnpm-lock.yaml ]; then \
        npm install -g pnpm && pnpm install --frozen-lockfile; \
    elif [ -f yarn.lock ]; then \
        yarn install --frozen-lockfile; \
    else \
        npm ci || npm install; \
    fi

# Copy source and build
COPY . .

# Build-time env vars for API URL (can override at build)
ARG VITE_API_URL=http://localhost:8000
ARG VITE_WS_URL=ws://localhost:8000/ws
ARG VITE_GO2RTC_URL=http://localhost:1984

ENV VITE_API_URL=${VITE_API_URL}
ENV VITE_WS_URL=${VITE_WS_URL}
ENV VITE_GO2RTC_URL=${VITE_GO2RTC_URL}

RUN npm run build

# ── Stage 2: Serve with Nginx ────────────────────────────────
FROM nginx:alpine AS production

# Remove default config
RUN rm /etc/nginx/conf.d/default.conf

# Copy custom nginx config
COPY nginx.conf /etc/nginx/conf.d/race-vision.conf

# Copy built app
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD wget -qO- http://localhost:80/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
