# syntax=docker/dockerfile:1.7
# Build stage
FROM node:22-alpine AS builder

WORKDIR /app

# Copy package files
COPY package.json package-lock.json* ./

# Install dependencies
RUN --mount=type=secret,id=custom_ca,required=false \
    if [ -s /run/secrets/custom_ca ]; then \
      export NODE_EXTRA_CA_CERTS=/run/secrets/custom_ca; \
    fi; \
    npm ci --no-audit --no-fund

# Copy source files
COPY . .

# Build the application
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built files to nginx
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy nginx config as template (envsubst substitutes ${NGINX_RESOLVER} at container startup)
COPY nginx.conf /etc/nginx/templates/default.conf.template

# Default DNS resolver: Docker's embedded DNS (works for Docker Compose / standalone Docker)
# Override via env var for other environments (e.g. Kubernetes sets kube-dns)
ENV NGINX_RESOLVER=127.0.0.11

# Default backend address: Docker Compose service name
# Override for K8s with FQDN (e.g. backend.tops.svc.cluster.local:8010)
ENV NGINX_BACKEND=backend:8010

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
