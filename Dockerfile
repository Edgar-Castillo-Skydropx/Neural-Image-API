# Dockerfile para la API de reconocimiento de imágenes con red neuronal

# Usar una imagen base oficial de Node.js
FROM node:20-alpine

# Crear directorio de la aplicación
WORKDIR /usr/src/app

# Instalar dependencias de sistema necesarias
RUN apk add --no-cache \
    python3 \
    make \
    g++ \
    jpeg-dev \
    cairo-dev \
    pango-dev \
    giflib-dev

# Copiar archivos de definición de dependencias
COPY package*.json ./

# Instalar dependencias
RUN pnpm install

# Copiar el código fuente
COPY . .

# Compilar TypeScript a JavaScript
RUN pnpm build

# Crear directorios necesarios
RUN mkdir -p uploads

# Exponer el puerto definido en las variables de entorno
EXPOSE ${PORT:-3000}

# Comando para iniciar la aplicación
CMD ["node", "dist/index.js"]
