# Guion para Video Tutorial: API de Reconocimiento de Imágenes con Red Neuronal en TypeScript

## Introducción (0:00 - 2:00)

Hola y bienvenidos a este tutorial completo sobre nuestra API de reconocimiento de imágenes con red neuronal implementada en TypeScript. En este video, vamos a explorar paso a paso cómo funciona esta API, desde los conceptos más básicos hasta los detalles más complejos de su implementación.

Lo que hace especial a este proyecto es que hemos desarrollado una red neuronal desde cero, aplicando fundamentos matemáticos avanzados sin utilizar librerías externas de machine learning. Todo está completamente tipado con TypeScript, aplicando principios de programación orientada a objetos como herencia y polimorfismo.

### Lo que aprenderás en este video:
1. Estructura general del proyecto y arquitectura
2. Fundamentos matemáticos de la red neuronal
3. Implementación de capas, activaciones y optimizadores
4. Endpoints de la API y cómo utilizarlos
5. Sistema de entrenamiento y almacenamiento de pesos
6. Despliegue con Docker y Docker Compose
7. Ejemplos prácticos de uso

## Parte 1: Estructura del Proyecto y Arquitectura (2:00 - 5:00)

Comencemos explorando la estructura del proyecto. Nuestra API sigue una arquitectura modular y bien organizada:

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
```

Una característica importante de nuestro proyecto es el uso de path aliases para mejorar la legibilidad del código. En lugar de usar rutas relativas complejas, utilizamos aliases como `@/api`, `@/db`, `@/neural`, etc.

Por ejemplo, en lugar de escribir:
```typescript
import { ImageService } from '../../api/services/ImageService';
```

Escribimos:
```typescript
import { ImageService } from '@/api/services/ImageService';
```

Esto hace que el código sea más limpio y mantenible.

## Parte 2: Fundamentos Matemáticos de la Red Neuronal (5:00 - 10:00)

Ahora, vamos a adentrarnos en los fundamentos matemáticos de nuestra red neuronal. La base de cualquier red neuronal son las operaciones matriciales, por lo que hemos implementado una clase `Matrix` que maneja estas operaciones.

Veamos el archivo `src/neural/math/Matrix.ts`:

```typescript
export class Matrix implements IMatrix {
  private data: number[][];

  constructor(rows: number, cols: number, initialValue: number = 0) {
    this.data = Array(rows).fill(0).map(() => Array(cols).fill(initialValue));
  }

  // Operaciones básicas
  public add(other: Matrix): Matrix {
    // Implementación de suma matricial
  }

  public subtract(other: Matrix): Matrix {
    // Implementación de resta matricial
  }

  public multiply(other: Matrix): Matrix {
    // Implementación de multiplicación matricial
  }

  public hadamardProduct(other: Matrix): Matrix {
    // Implementación de producto elemento a elemento (Hadamard)
  }

  public transpose(): Matrix {
    // Implementación de transposición
  }
}
```

Estas operaciones son fundamentales para implementar:

1. **Propagación hacia adelante (Forward Propagation)**: Proceso por el cual la información fluye desde la entrada hasta la salida de la red.
2. **Retropropagación (Backpropagation)**: Algoritmo para ajustar los pesos de la red basado en el error de predicción.

## Parte 3: Implementación de Capas, Activaciones y Optimizadores (10:00 - 18:00)

### Funciones de Activación

Las funciones de activación son cruciales en las redes neuronales, ya que introducen no-linealidad al sistema. Hemos implementado varias funciones de activación comunes:

```typescript
// BaseActivation.ts
export abstract class BaseActivation implements IActivation {
  public abstract forward(input: number[][]): number[][];
  public abstract backward(gradient: number[][]): number[][];
}

// Sigmoid.ts
export class Sigmoid extends BaseActivation {
  public forward(input: number[][]): number[][] {
    // Implementación de la función sigmoide: 1 / (1 + e^(-x))
  }

  public backward(gradient: number[][]): number[][] {
    // Derivada de la función sigmoide: sigmoid(x) * (1 - sigmoid(x))
  }
}
```

También hemos implementado ReLU y Tanh siguiendo el mismo patrón.

### Capas Neuronales

Las capas son los bloques de construcción de nuestra red neuronal. Hemos implementado diferentes tipos:

```typescript
// BaseLayer.ts
export abstract class BaseLayer implements ILayer {
  protected id: string;
  protected type: string;
  protected weights: Matrix | null = null;
  protected bias: Matrix | null = null;
  protected activation: IActivation | null = null;
  protected lastInput: number[][] | null = null;
  protected lastOutput: number[][] | null = null;

  constructor(id: string, type: string) {
    this.id = id;
    this.type = type;
  }

  public abstract forward(input: number[][]): number[][];
  public abstract backward(gradient: number[][], learningRate: number): number[][];
}

// DenseLayer.ts
export class DenseLayer extends BaseLayer {
  constructor(
    id: string,
    private inputSize: number,
    private outputSize: number,
    activationType: ActivationType
  ) {
    super(id, 'dense');
    this.weights = new Matrix(inputSize, outputSize).randomize();
    this.bias = new Matrix(1, outputSize).randomize();
    this.activation = ActivationFactory.create(activationType);
  }

  public forward(input: number[][]): number[][] {
    // Implementación de propagación hacia adelante
  }

  public backward(gradient: number[][], learningRate: number): number[][] {
    // Implementación de retropropagación
  }
}
```

### Optimizadores

Los optimizadores son algoritmos que ajustan los pesos de la red para minimizar la función de pérdida:

```typescript
// SGDOptimizer.ts
export class SGDOptimizer extends BaseOptimizer {
  constructor(learningRate: number = 0.01) {
    super('sgd', learningRate);
  }

  public updateWeights(weights: number[][], gradients: number[][]): number[][] {
    // Implementación del descenso de gradiente estocástico
    // w = w - lr * gradient
  }
}
```

## Parte 4: Endpoints de la API y Cómo Utilizarlos (18:00 - 25:00)

Nuestra API expone varios endpoints para interactuar con la red neuronal:

### Clasificación de Imágenes

```typescript
// imageRoutes.ts
router.post(
  '/classify',
  upload.single('image'),
  handleMulterError,
  validateImageUpload,
  imageController.classifyImage
);

router.get('/status', imageController.getStatus);
```

Para clasificar una imagen, envías una solicitud POST a `/api/images/classify` con la imagen como un archivo multipart:

```bash
curl -X POST -F "image=@/ruta/a/imagen.jpg" http://localhost:3000/api/images/classify
```

La respuesta será algo como:

```json
{
  "success": true,
  "data": {
    "classification": "gato",
    "confidence": 0.95,
    "processingTime": 0.123
  }
}
```

### Entrenamiento

```typescript
// trainingRoutes.ts
router.post(
  '/train',
  upload.array('images', 50),
  handleMulterError,
  trainingController.trainNetwork
);

router.get('/status', trainingController.getTrainingStatus);
router.post('/save', trainingController.saveModel);
router.post('/load', trainingController.loadModel);
```

Para entrenar la red, envías una solicitud POST a `/api/training/train` con las imágenes y sus etiquetas:

```bash
curl -X POST \
  -F "images=@/ruta/a/imagen1.jpg" \
  -F "images=@/ruta/a/imagen2.jpg" \
  -F "labels=[\"gato\",\"perro\"]" \
  -F "epochs=10" \
  -F "learningRate=0.01" \
  http://localhost:3000/api/training/train
```

## Parte 5: Sistema de Entrenamiento y Almacenamiento de Pesos (25:00 - 32:00)

El entrenamiento de la red neuronal es un proceso que puede llevar tiempo, por lo que lo hemos implementado de forma asíncrona:

```typescript
// TrainingService.ts
public async startTraining(options: {
  imagePaths: string[];
  labels: string[];
  epochs?: number;
  learningRate?: number;
  batchSize?: number;
}): Promise<string> {
  // Crear registro de entrenamiento en la base de datos
  const trainingData = await this.trainingRepository.createTraining({...});
  
  const trainingId = trainingData._id.toString();
  
  // Crear modelo para el entrenamiento
  const model = new SequentialModel(`model_${trainingId}`, `Model_${Date.now()}`);
  
  // Iniciar el entrenamiento en segundo plano
  this.runTraining(
    trainingId,
    model,
    options.imagePaths,
    options.labels,
    options.epochs || 10,
    options.batchSize || 32,
    options.learningRate || 0.01
  );
  
  return trainingId;
}
```

Los pesos y configuraciones de los modelos entrenados se almacenan en MongoDB:

```typescript
// NetworkModel.ts
export interface INetworkModelDocument extends Document {
  name: string;
  description?: string;
  architecture: string;
  layers: {
    id: string;
    type: string;
    weights: Record<string, any>;
  }[];
  performance: {
    accuracy?: number;
    loss?: number;
    validationAccuracy?: number;
    validationLoss?: number;
  };
  metadata: {
    createdAt: Date;
    updatedAt: Date;
    version: string;
    trainingTime?: number;
    epochs?: number;
  };
}
```

## Parte 6: Despliegue con Docker y Docker Compose (32:00 - 38:00)

Para facilitar el despliegue, hemos configurado Docker y Docker Compose:

### Dockerfile

```dockerfile
FROM node:20-alpine

WORKDIR /usr/src/app

RUN apk add --no-cache \
    python3 \
    make \
    g++ \
    jpeg-dev \
    cairo-dev \
    pango-dev \
    giflib-dev

COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

RUN mkdir -p uploads

EXPOSE ${PORT:-3000}
CMD ["node", "dist/index.js"]
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: neural-image-api
    restart: unless-stopped
    ports:
      - "${PORT:-3000}:${PORT:-3000}"
    volumes:
      - ./uploads:/usr/src/app/uploads
    environment:
      - NODE_ENV=${NODE_ENV:-development}
      - PORT=${PORT:-3000}
      - MONGODB_URI=mongodb://${MONGO_USER:-user}:${MONGO_PASSWORD:-password}@mongodb:27017/${MONGO_DB:-neural_image_db}?authSource=admin
    depends_on:
      - mongodb
    networks:
      - neural-network

  mongodb:
    image: mongo:latest
    container_name: neural-image-mongodb
    restart: unless-stopped
    environment:
      - MONGO_INITDB_ROOT_USERNAME=${MONGO_USER:-user}
      - MONGO_INITDB_ROOT_PASSWORD=${MONGO_PASSWORD:-password}
      - MONGO_INITDB_DATABASE=${MONGO_DB:-neural_image_db}
    volumes:
      - mongodb_data:/data/db
    ports:
      - "${MONGO_PORT:-27017}:27017"
    networks:
      - neural-network

volumes:
  mongodb_data:
    driver: local

networks:
  neural-network:
    driver: bridge
```

Para desplegar la aplicación:

```bash
docker-compose up -d
```

## Parte 7: Ejemplos Prácticos de Uso (38:00 - 45:00)

Vamos a ver algunos ejemplos prácticos de cómo utilizar la API:

### Ejemplo 1: Clasificar una imagen

```javascript
// Ejemplo con JavaScript y Fetch API
async function classifyImage(imageFile) {
  const formData = new FormData();
  formData.append('image', imageFile);
  
  const response = await fetch('http://localhost:3000/api/images/classify', {
    method: 'POST',
    body: formData
  });
  
  const result = await response.json();
  console.log('Clasificación:', result.data.classification);
  console.log('Confianza:', result.data.confidence);
}

// Uso
const fileInput = document.getElementById('imageInput');
fileInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  classifyImage(file);
});
```

### Ejemplo 2: Entrenar la red con nuevas imágenes

```javascript
async function trainNetwork(imageFiles, labels) {
  const formData = new FormData();
  
  imageFiles.forEach(file => {
    formData.append('images', file);
  });
  
  formData.append('labels', JSON.stringify(labels));
  formData.append('epochs', '20');
  formData.append('learningRate', '0.005');
  
  const response = await fetch('http://localhost:3000/api/training/train', {
    method: 'POST',
    body: formData
  });
  
  const result = await response.json();
  const trainingId = result.data.trainingId;
  
  // Monitorear el progreso del entrenamiento
  monitorTraining(trainingId);
}

async function monitorTraining(trainingId) {
  const interval = setInterval(async () => {
    const response = await fetch(`http://localhost:3000/api/training/status?trainingId=${trainingId}`);
    const result = await response.json();
    
    console.log(`Progreso: ${result.data.progress}%`);
    
    if (result.data.status === 'completed' || result.data.status === 'failed') {
      clearInterval(interval);
      console.log('Entrenamiento finalizado:', result.data.status);
    }
  }, 1000);
}
```

## Conclusión (45:00 - 48:00)

En este tutorial, hemos explorado en detalle nuestra API de reconocimiento de imágenes con red neuronal implementada en TypeScript. Hemos visto:

1. La estructura y arquitectura del proyecto
2. Los fundamentos matemáticos de la red neuronal
3. La implementación de capas, activaciones y optimizadores
4. Los endpoints de la API y cómo utilizarlos
5. El sistema de entrenamiento y almacenamiento de pesos
6. El despliegue con Docker y Docker Compose
7. Ejemplos prácticos de uso

Lo que hace especial a este proyecto es que hemos implementado todo desde cero, sin depender de librerías externas de machine learning, y con un tipado completo en TypeScript.

Espero que este tutorial te haya sido útil para entender cómo funciona nuestra API y cómo puedes utilizarla en tus propios proyectos. Si tienes alguna pregunta o comentario, no dudes en contactarnos.

¡Gracias por ver este tutorial!
