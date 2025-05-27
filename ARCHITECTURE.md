# Diseño de Arquitectura - API de Reconocimiento de Imágenes con Red Neuronal

## Arquitectura General

```
┌─────────────────────────────────────────────────────────────┐
│                      Cliente HTTP                           │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      API REST (Express)                     │
│                                                             │
│  ┌─────────────────┐    ┌─────────────────┐                 │
│  │   Controllers   │───▶│    Services     │                 │
│  └─────────────────┘    └────────┬────────┘                 │
│                                  │                          │
│                                  ▼                          │
│  ┌─────────────────┐    ┌─────────────────┐                 │
│  │     Models      │◀───│   Repositories  │                 │
│  └─────────────────┘    └─────────────────┘                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Red Neuronal                             │
│                                                             │
│  ┌─────────────────┐    ┌─────────────────┐                 │
│  │     Layers      │───▶│   Activations   │                 │
│  └─────────────────┘    └─────────────────┘                 │
│                                                             │
│  ┌─────────────────┐    ┌─────────────────┐                 │
│  │   Optimizers    │◀───│     Matrix      │                 │
│  └─────────────────┘    └─────────────────┘                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Base de Datos (MongoDB)                   │
└─────────────────────────────────────────────────────────────┘
```

## Estructura de Directorios

```
neural-image-api/
├── src/
│   ├── api/
│   │   ├── controllers/
│   │   │   ├── ImageController.ts
│   │   │   └── TrainingController.ts
│   │   ├── middlewares/
│   │   │   ├── errorHandler.ts
│   │   │   ├── imageUpload.ts
│   │   │   └── validation.ts
│   │   ├── routes/
│   │   │   ├── imageRoutes.ts
│   │   │   └── trainingRoutes.ts
│   │   └── services/
│   │       ├── ImageService.ts
│   │       └── TrainingService.ts
│   ├── config/
│   │   ├── database.ts
│   │   ├── environment.ts
│   │   └── server.ts
│   ├── core/
│   │   ├── interfaces/
│   │   │   ├── IActivation.ts
│   │   │   ├── ILayer.ts
│   │   │   ├── IMatrix.ts
│   │   │   ├── IModel.ts
│   │   │   ├── IOptimizer.ts
│   │   │   └── ITrainable.ts
│   │   ├── types/
│   │   │   ├── ActivationType.ts
│   │   │   ├── LayerType.ts
│   │   │   ├── MatrixType.ts
│   │   │   └── OptimizerType.ts
│   │   └── utils/
│   │       ├── imageProcessing.ts
│   │       └── mathUtils.ts
│   ├── db/
│   │   ├── models/
│   │   │   ├── NetworkModel.ts
│   │   │   └── TrainingData.ts
│   │   └── repositories/
│   │       ├── ModelRepository.ts
│   │       └── TrainingRepository.ts
│   ├── neural/
│   │   ├── activations/
│   │   │   ├── ActivationFactory.ts
│   │   │   ├── BaseActivation.ts
│   │   │   ├── ReLU.ts
│   │   │   ├── Sigmoid.ts
│   │   │   └── Tanh.ts
│   │   ├── layers/
│   │   │   ├── BaseLayer.ts
│   │   │   ├── ConvolutionalLayer.ts
│   │   │   ├── DenseLayer.ts
│   │   │   ├── InputLayer.ts
│   │   │   ├── LayerFactory.ts
│   │   │   └── PoolingLayer.ts
│   │   ├── math/
│   │   │   ├── Matrix.ts
│   │   │   └── Vector.ts
│   │   ├── models/
│   │   │   ├── BaseModel.ts
│   │   │   ├── ConvolutionalNetwork.ts
│   │   │   └── SequentialModel.ts
│   │   ├── optimizers/
│   │   │   ├── AdamOptimizer.ts
│   │   │   ├── BaseOptimizer.ts
│   │   │   ├── OptimizerFactory.ts
│   │   │   └── SGDOptimizer.ts
│   │   └── utils/
│   │       ├── imagePreprocessing.ts
│   │       └── modelSerializer.ts
│   ├── app.ts
│   └── index.ts
├── tests/
│   ├── api/
│   │   └── controllers/
│   ├── neural/
│   │   ├── activations/
│   │   ├── layers/
│   │   └── math/
│   └── utils/
├── .dockerignore
├── .env.example
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── package.json
├── README.md
├── REQUIREMENTS.md
├── todo.md
└── tsconfig.json
```

## Diseño de la Red Neuronal

### Jerarquía de Clases e Interfaces

```
┌───────────────────┐
│   <<interface>>   │
│     IMatrix       │
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│      Matrix       │
└───────────────────┘

┌───────────────────┐
│   <<interface>>   │
│   IActivation     │
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│  BaseActivation   │◄─────┐
└───────────────────┘      │
        ▲                  │
        │                  │
┌───────┴───────┬──────────┴───────┬───────────────┐
│    Sigmoid    │      ReLU        │     Tanh      │
└───────────────┴──────────────────┴───────────────┘

┌───────────────────┐
│   <<interface>>   │
│      ILayer       │
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│    BaseLayer      │◄─────┐
└───────────────────┘      │
        ▲                  │
        │                  │
┌───────┴───────┬──────────┴───────┬───────────────┐
│   InputLayer  │    DenseLayer    │ ConvLayer     │
└───────────────┴──────────────────┴───────┬───────┘
                                           │
                                           ▼
                                   ┌───────────────┐
                                   │  PoolingLayer │
                                   └───────────────┘

┌───────────────────┐
│   <<interface>>   │
│    IOptimizer     │
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│   BaseOptimizer   │◄─────┐
└───────────────────┘      │
        ▲                  │
        │                  │
┌───────┴───────┬──────────┴───────┐
│ SGDOptimizer  │  AdamOptimizer   │
└───────────────┴──────────────────┘

┌───────────────────┐
│   <<interface>>   │
│      IModel       │
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│    BaseModel      │◄─────┐
└───────────────────┘      │
        ▲                  │
        │                  │
┌───────┴───────┬──────────┴───────┐
│SequentialModel│ ConvolutionalNet │
└───────────────┴──────────────────┘
```

## Flujo de Datos

1. **Recepción de Imágenes**:
   - Cliente envía imagen vía HTTP POST
   - Middleware procesa y valida la imagen
   - Controller recibe la petición y la pasa al Service

2. **Procesamiento de Imágenes**:
   - ImageService preprocesa la imagen (normalización, redimensionamiento)
   - Conversión a formato matricial para la red neuronal

3. **Clasificación**:
   - Carga del modelo entrenado desde la base de datos
   - Propagación hacia adelante a través de la red neuronal
   - Obtención de predicciones/clasificaciones

4. **Respuesta**:
   - Formateo de resultados
   - Envío de respuesta HTTP con la clasificación

5. **Entrenamiento** (flujo separado):
   - Recepción de datos de entrenamiento
   - Preprocesamiento de imágenes
   - Entrenamiento de la red (propagación hacia adelante, cálculo de error, retropropagación)
   - Actualización de pesos mediante optimizador
   - Almacenamiento del modelo entrenado en la base de datos

## Componentes Clave de la Red Neuronal

1. **Matrix**: Implementación matemática para operaciones matriciales
   - Multiplicación, suma, transposición, etc.
   - Operaciones elemento a elemento

2. **Activations**: Funciones de activación y sus derivadas
   - Sigmoid: σ(x) = 1/(1+e^(-x))
   - ReLU: f(x) = max(0, x)
   - Tanh: tanh(x) = (e^x - e^(-x))/(e^x + e^(-x))

3. **Layers**: Capas de la red neuronal
   - Input: Capa de entrada
   - Dense: Capa completamente conectada
   - Convolutional: Capa convolucional para procesamiento de imágenes
   - Pooling: Capa de agrupamiento para reducción de dimensionalidad

4. **Optimizers**: Algoritmos de optimización
   - SGD: Descenso de gradiente estocástico
   - Adam: Optimización adaptativa con estimación de momento

5. **Models**: Arquitecturas de red neuronal
   - Sequential: Capas apiladas secuencialmente
   - Convolutional: Red convolucional especializada en imágenes

## Integración con Docker

- **API Container**: Servicio Node.js con la API y la red neuronal
- **MongoDB Container**: Base de datos para almacenamiento de modelos y datos de entrenamiento
- **Variables de entorno**: Configuración mediante archivo .env

## Consideraciones de Rendimiento

- Implementación eficiente de operaciones matriciales
- Uso de técnicas de paralelización cuando sea posible
- Optimización de memoria para grandes conjuntos de datos
- Estrategias de caché para modelos frecuentemente utilizados
