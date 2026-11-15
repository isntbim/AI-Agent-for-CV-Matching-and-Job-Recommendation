# The Next.js front end (frontend/). The browser calls /api/* on this server, which forwards it to the API container
# (next.config.js rewrites), so the page works from any host address without CORS or a baked-in API URL.
# Built by docker/docker-compose.yml (service "frontend") and by CI (ghcr.io/.../frontend).
FROM node:22.23.4-alpine AS build
WORKDIR /app
ENV NEXT_TELEMETRY_DISABLED=1 NEXT_PUBLIC_API_URL="" API_INTERNAL_URL=http://api:8000
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build && npm prune --omit=dev

FROM node:22.23.4-alpine
WORKDIR /app
ENV NODE_ENV=production NEXT_TELEMETRY_DISABLED=1 API_INTERNAL_URL=http://api:8000
COPY --from=build /app ./
EXPOSE 3000
CMD ["npm", "start"]
