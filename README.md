# API de Reconocimiento de Imágenes con Red Neuronal en TypeScript

Este proyecto implementa una API RESTful en Node.js con TypeScript que permite clasificar imágenes mediante una red neuronal implementada desde cero, sin utilizar librerías externas de machine learning.

## Características principales

- Red neuronal implementada desde cero con fundamentos matemáticos avanzados
- Tipado completo con TypeScript (interfaces, types, herencia, polimorfismo)
- API RESTful para subida y clasificación de imágenes
- Sistema de entrenamiento y almacenamiento de pesos en base de datos
- Contenerización con Docker y Docker Compose
- Configuración mediante variables de entorno
- Path aliases para importaciones más limpias (@/api, @/db, etc.)

## Estructura del proyecto

```
neural-image-api/
├── src/
│   ├── api/                  # Componentes de la API REST
│   │   ├── controllers/      # Controladores para endpoints
│   │   ├── middlewares/      # Middlewares (validación, errores)
│   │   ├── routes/           # Definición de rutas
│   │   └── services/         # Servicios de negocio
│   ├── config/               # Configuración de la aplicación
│   ├── core/                 # Interfaces y tipos base
│   │   ├── interfaces/       # Interfaces del sistema
│   │   └── types/            # Tipos y enumeraciones
│   ├── db/                   # Capa de acceso a datos
│   │   ├── models/           # Modelos de MongoDB
│   │   └── repositories/     # Repositorios para acceso a datos
│   ├── neural/               # Implementación de la red neuronal
│   │   ├── activations/      # Funciones de activación
│   │   ├── layers/           # Capas neuronales
│   │   ├── math/             # Operaciones matemáticas
│   │   ├── models/           # Modelos de red neuronal
│   │   └── optimizers/       # Algoritmos de optimización
│   ├── app.ts                # Configuración de Express
│   └── index.ts              # Punto de entrada
├── uploads/                  # Directorio para imágenes subidas
├── .dockerignore             # Archivos a ignorar en Docker
├── .env                      # Variables de entorno
├── .env.example              # Ejemplo de variables de entorno
├── docker-compose.yml        # Configuración de Docker Compose
├── Dockerfile                # Configuración de Docker
├── package.json              # Dependencias y scripts
└── tsconfig.json             # Configuración de TypeScript
```

## Tecnologías utilizadas

- **Node.js**: Plataforma de ejecución
- **TypeScript**: Lenguaje de programación tipado
- **Express**: Framework para API REST
- **MongoDB**: Base de datos para almacenamiento de modelos
- **Docker**: Contenerización de la aplicación
- **Docker Compose**: Orquestación de servicios

## Path Aliases

El proyecto utiliza path aliases para mejorar la legibilidad y mantenibilidad del código. Los aliases configurados son:

- `@/*`: Acceso a cualquier archivo desde la raíz de src/
- `@/api/*`: Acceso a componentes de la API (controllers, services, etc.)
- `@/core/*`: Acceso a interfaces y tipos base
- `@/db/*`: Acceso a modelos y repositorios de base de datos
- `@/neural/*`: Acceso a componentes de la red neuronal
- `@/config/*`: Acceso a archivos de configuración

Ejemplo de uso:

```typescript
// En lugar de:
import { ImageService } from "../../api/services/ImageService";

// Usar:
import { ImageService } from "@/api/services/ImageService";
```

## Implementación de la red neuronal

La red neuronal está implementada desde cero, sin utilizar librerías externas de machine learning. Incluye:

- **Operaciones matriciales**: Implementación de álgebra lineal para cálculos neuronales
- **Funciones de activación**: Sigmoid, ReLU, Tanh
- **Capas neuronales**: Dense, Convolutional, Input
- **Algoritmos de propagación**: Forward propagation, backpropagation
- **Optimizadores**: SGD (Stochastic Gradient Descent)

Todo el código está completamente tipado con TypeScript, utilizando interfaces, herencia y polimorfismo.

## Endpoints de la API

### Clasificación de imágenes

- `POST /api/images/classify`: Clasifica una imagen subida
- `GET /api/images/status`: Verifica el estado del servicio de clasificación

### Entrenamiento

- `POST /api/training/train`: Entrena la red neuronal con imágenes etiquetadas
- `GET /api/training/status`: Obtiene el estado del entrenamiento
- `POST /api/training/save`: Guarda el modelo entrenado
- `POST /api/training/load`: Carga un modelo previamente entrenado

## Configuración y despliegue

### Variables de entorno

La aplicación utiliza variables de entorno para su configuración. Crea un archivo `.env` basado en `.env.example`:

```
# Configuración del servidor
NODE_ENV=development
PORT=3000

# Configuración de MongoDB
MONGO_USER=user
MONGO_PASSWORD=password
MONGO_DB=neural_image_db
MONGO_PORT=27017

# Configuración de la red neuronal
DEFAULT_LEARNING_RATE=0.01
DEFAULT_BATCH_SIZE=32
DEFAULT_EPOCHS=10
MODEL_SAVE_PATH=./models

# Configuración de procesamiento de imágenes
MAX_IMAGE_SIZE=5242880  # 5MB en bytes
ALLOWED_IMAGE_TYPES=image/jpeg,image/png,image/gif
IMAGE_UPLOAD_PATH=./uploads
```

### Despliegue con Docker Compose

Para desplegar la aplicación con Docker Compose:

1. Asegúrate de tener Docker y Docker Compose instalados
2. Crea el archivo `.env` con la configuración deseada
3. Ejecuta el siguiente comando:

```bash
docker-compose up -d
```

Esto iniciará tanto la API como la base de datos MongoDB en contenedores separados.

## Desarrollo

Para ejecutar la aplicación en modo desarrollo:

1. Instala las dependencias:

```bash
pnpm install
```

2. Inicia el servidor de desarrollo:

```bash
pnpm dev
```

## Compilación

Para compilar el proyecto a JavaScript:

```bash
pnpm build
```

El código compilado se generará en el directorio `dist/`.

## Ejecución

Para ejecutar la aplicación compilada:

```bash
pnpm start
```
