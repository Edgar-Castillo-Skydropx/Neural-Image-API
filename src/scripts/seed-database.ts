/**
 * Script para poblar la base de datos de la API de reconocimiento de imágenes
 * Este script crea modelos de ejemplo y datos de entrenamiento para pruebas
 */

import mongoose from "mongoose";
import dotenv from "dotenv";
import path from "path";

// Importar modelos
import { NetworkModel } from "../db/models/NetworkModel";
import { TrainingData } from "../db/models/TrainingData";

// Cargar variables de entorno
dotenv.config({ path: path.resolve(__dirname, "../.env") });

// Función para generar una matriz aleatoria
const generateRandomMatrix = (rows: number, cols: number): number[][] => {
  return Array(rows)
    .fill(0)
    .map(() =>
      Array(cols)
        .fill(0)
        .map(() => Math.random() * 2 - 1)
    );
};

// Función para crear modelos de ejemplo
const createSampleModels = async () => {
  console.log("Creando modelos de ejemplo...");

  // Clases para el modelo básico (MNIST)
  const basicClasses = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];

  // Clases para el modelo avanzado (CIFAR-10)
  const advancedClasses = [
    "avión",
    "automóvil",
    "pájaro",
    "gato",
    "ciervo",
    "perro",
    "rana",
    "caballo",
    "barco",
    "camión",
  ];

  // Modelo 1: Clasificador básico
  const basicModel = new NetworkModel({
    name: "Clasificador Básico",
    description: "Modelo básico para clasificación de imágenes",
    architecture: "sequential",
    layers: [
      {
        id: "input_1",
        type: "input",
        weights: {},
      },
      {
        id: "dense_1",
        type: "dense",
        weights: {
          weights: generateRandomMatrix(784, 128),
          bias: generateRandomMatrix(1, 128),
        },
      },
      {
        id: "dense_2",
        type: "dense",
        weights: {
          weights: generateRandomMatrix(128, 64),
          bias: generateRandomMatrix(1, 64),
        },
      },
      {
        id: "dense_3",
        type: "dense",
        weights: {
          weights: generateRandomMatrix(64, 10),
          bias: generateRandomMatrix(1, 10),
        },
      },
    ],
    performance: {
      accuracy: 0.85,
      loss: 0.15,
      validationAccuracy: 0.82,
      validationLoss: 0.18,
    },
    metadata: {
      createdAt: new Date(),
      updatedAt: new Date(),
      version: "1.0.0",
      trainingTime: 120,
      epochs: 10,
      classes: basicClasses,
      imageSize: 28, // MNIST usa imágenes de 28x28
    },
  });

  // Modelo 2: Clasificador avanzado
  const advancedModel = new NetworkModel({
    name: "Clasificador Avanzado",
    description:
      "Modelo avanzado con capas convolucionales para clasificación de imágenes",
    architecture: "convolutional",
    layers: [
      {
        id: "input_1",
        type: "input",
        weights: {},
      },
      {
        id: "conv_1",
        type: "convolutional",
        weights: {
          filters: generateRandomMatrix(3, 3).map((row) =>
            row.map(() => generateRandomMatrix(3, 3))
          ),
          bias: generateRandomMatrix(1, 32),
        },
      },
      {
        id: "dense_1",
        type: "dense",
        weights: {
          weights: generateRandomMatrix(1568, 128),
          bias: generateRandomMatrix(1, 128),
        },
      },
      {
        id: "dense_2",
        type: "dense",
        weights: {
          weights: generateRandomMatrix(128, 10),
          bias: generateRandomMatrix(1, 10),
        },
      },
    ],
    performance: {
      accuracy: 0.92,
      loss: 0.08,
      validationAccuracy: 0.9,
      validationLoss: 0.1,
    },
    metadata: {
      createdAt: new Date(Date.now() - 86400000), // 1 día atrás
      updatedAt: new Date(),
      version: "1.1.0",
      trainingTime: 300,
      epochs: 20,
      classes: advancedClasses,
      imageSize: 32, // CIFAR-10 usa imágenes de 32x32
    },
  });

  await basicModel.save();
  await advancedModel.save();

  console.log("Modelos de ejemplo creados correctamente");
};

// Función para crear datos de entrenamiento de ejemplo
const createSampleTrainingData = async () => {
  console.log("Creando datos de entrenamiento de ejemplo...");

  // Entrenamiento 1: Completado
  const completedTraining = new TrainingData({
    name: "Entrenamiento Completo",
    description: "Entrenamiento de prueba completado",
    status: "completed",
    progress: 100,
    images: [
      {
        path: "/Users/skydropx/Desktop/machine-learning/neural-image-api/uploads/gato.jpg",
        label: "gato",
        processed: true,
      },
      {
        path: "/Users/skydropx/Desktop/machine-learning/neural-image-api/uploads/perro.png",
        label: "perro",
        processed: true,
      },
      {
        path: "/Users/skydropx/Desktop/machine-learning/neural-image-api/uploads/pajaro.png",
        label: "pájaro",
        processed: true,
      },
    ],
    configuration: {
      epochs: 10,
      batchSize: 32,
      learningRate: 0.01,
      optimizer: "sgd",
      imageSize: 32, // Tamaño de imagen para el entrenamiento
    },
    results: {
      accuracy: [0.5, 0.6, 0.7, 0.75, 0.8, 0.82, 0.85, 0.87, 0.88, 0.9],
      loss: [0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.18, 0.15, 0.12, 0.1],
      validationAccuracy: [
        0.45, 0.55, 0.65, 0.7, 0.75, 0.78, 0.8, 0.82, 0.84, 0.85,
      ],
      validationLoss: [0.55, 0.45, 0.4, 0.35, 0.3, 0.25, 0.22, 0.2, 0.18, 0.15],
    },
    metadata: {
      startTime: new Date(Date.now() - 3600000), // 1 hora atrás
      endTime: new Date(Date.now() - 1800000), // 30 minutos atrás
      duration: 1800, // 30 minutos en segundos
      modelId: "60f1b5b5b5b5b5b5b5b5b5b5", // ID ficticio, se actualizará después
    },
  });

  // Entrenamiento 2: En progreso
  const inProgressTraining = new TrainingData({
    name: "Entrenamiento En Progreso",
    description: "Entrenamiento de prueba en progreso",
    status: "in_progress",
    progress: 60,
    images: [
      {
        path: "/Users/skydropx/Desktop/machine-learning/neural-image-api/uploads/carro.jpg",
        label: "coche",
        processed: true,
      },
      {
        path: "/Users/skydropx/Desktop/machine-learning/neural-image-api/uploads/bici.jpg",
        label: "bicicleta",
        processed: true,
      },
      {
        path: "/Users/skydropx/Desktop/machine-learning/neural-image-api/uploads/avion.jpg",
        label: "avión",
        processed: false,
      },
    ],
    configuration: {
      epochs: 20,
      batchSize: 16,
      learningRate: 0.005,
      optimizer: "sgd",
      imageSize: 64, // Tamaño de imagen más grande para este entrenamiento
    },
    results: {
      accuracy: [
        0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.72, 0.75, 0.78, 0.8, 0.82, 0.83,
      ],
      loss: [0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.28, 0.25, 0.22, 0.2, 0.18, 0.17],
      validationAccuracy: [
        0.35, 0.45, 0.5, 0.55, 0.6, 0.65, 0.68, 0.7, 0.72, 0.75, 0.77, 0.78,
      ],
      validationLoss: [
        0.65, 0.55, 0.5, 0.45, 0.4, 0.35, 0.32, 0.3, 0.28, 0.25, 0.23, 0.22,
      ],
    },
    metadata: {
      startTime: new Date(Date.now() - 1200000), // 20 minutos atrás
      endTime: null,
      duration: null,
      modelId: null,
    },
  });

  // Entrenamiento 3: Pendiente
  const pendingTraining = new TrainingData({
    name: "Entrenamiento Pendiente",
    description: "Entrenamiento de prueba pendiente",
    status: "pending",
    progress: 0,
    images: [
      {
        path: "/Users/skydropx/Desktop/machine-learning/neural-image-api/uploads/manzana.jpg",
        label: "manzana",
        processed: false,
      },
      {
        path: "/Users/skydropx/Desktop/machine-learning/neural-image-api/uploads/platano.jpg",
        label: "plátano",
        processed: false,
      },
    ],
    configuration: {
      epochs: 15,
      batchSize: 32,
      learningRate: 0.01,
      optimizer: "sgd",
      imageSize: 128, // Tamaño de imagen grande para este entrenamiento
    },
    results: {
      accuracy: [],
      loss: [],
      validationAccuracy: [],
      validationLoss: [],
    },
    metadata: {
      startTime: null,
      endTime: null,
      duration: null,
      modelId: null,
    },
  });

  await completedTraining.save();
  await inProgressTraining.save();
  await pendingTraining.save();

  console.log("Datos de entrenamiento de ejemplo creados correctamente");
};

// Función principal para poblar la base de datos
const seedDatabase = async () => {
  try {
    // Conectar a MongoDB
    const mongoUri =
      process.env.MONGODB_URI || "mongodb://localhost:27017/neural-image-api";
    console.log(`Conectando a MongoDB: ${mongoUri}`);

    await mongoose.connect(mongoUri);
    console.log("Conexión a MongoDB establecida correctamente");

    // Limpiar colecciones existentes
    console.log("Limpiando colecciones existentes...");
    await NetworkModel.deleteMany({});
    await TrainingData.deleteMany({});

    // Crear datos de ejemplo
    await createSampleModels();
    await createSampleTrainingData();

    // Actualizar referencias entre modelos y entrenamientos
    console.log("Actualizando referencias entre modelos y entrenamientos...");

    const models = await NetworkModel.find().sort({ "metadata.createdAt": 1 });
    const trainings = await TrainingData.find({ status: "completed" });

    if (models.length > 0 && trainings.length > 0) {
      trainings[0].metadata.modelId = models[0]._id;
      await trainings[0].save();
      console.log("Referencias actualizadas correctamente");
    }

    console.log("Base de datos poblada correctamente");
  } catch (error) {
    console.error("Error al poblar la base de datos:", error);
  } finally {
    // Cerrar conexión
    await mongoose.disconnect();
    console.log("Conexión a MongoDB cerrada");
  }
};

// Ejecutar script
seedDatabase();
