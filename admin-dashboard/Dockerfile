FROM node:18-bullseye

WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm ci

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host"]
