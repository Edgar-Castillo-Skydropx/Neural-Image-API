/**
 * Script de validación para la red neuronal de clasificación de imágenes
 *
 * Este script crea un modelo de red neuronal, lo entrena con datos sintéticos
 * y valida su funcionamiento para asegurar que la clasificación es correcta.
 */

import { SequentialModel } from "@/neural/models/SequentialModel";
import { LayerType } from "@/neural/layers/LayerFactory";
import { ActivationType } from "@/core/types/ActivationType";
import { SGDOptimizer } from "@/neural/optimizers/SGDOptimizer";
import fs from "fs";
import path from "path";

// Clases para clasificación (CIFAR-10)
const CLASSES = [
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

/**
 * Genera datos sintéticos para entrenamiento y validación
 * @param numSamples Número de muestras a generar
 * @param numClasses Número de clases
 * @param inputSize Tamaño de entrada
 * @returns Datos de entrada y salida esperada
 */
function generateSyntheticData(
  numSamples: number,
  numClasses: number,
  inputSize: number
): {
  inputs: number[][][];
  targets: number[][][];
} {
  const inputs: number[][][] = [];
  const targets: number[][][] = [];

  for (let i = 0; i < numSamples; i++) {
    // Generar clase aleatoria
    const classIndex = Math.floor(Math.random() * numClasses);

    // Generar datos de entrada con sesgo hacia la clase correcta
    const input: number[] = [];
    for (let j = 0; j < inputSize; j++) {
      // Añadir ruido aleatorio
      input.push(Math.random() * 0.5);
    }

    // Añadir señal más fuerte para la clase correcta
    for (let j = 0; j < 10; j++) {
      const pos = classIndex * 10 + j;
      if (pos < inputSize) {
        input[pos] += 0.5 + Math.random() * 0.5; // Valor más alto para la clase correcta
      }
    }

    // Crear etiqueta one-hot
    const target: number[] = Array(numClasses).fill(0);
    target[classIndex] = 1;

    inputs.push([input]);
    targets.push([target]);
  }

  return { inputs, targets };
}

/**
 * Crea y configura un modelo de clasificación
 * @param inputSize Tamaño de entrada
 * @param hiddenSize Tamaño de la capa oculta
 * @param numClasses Número de clases
 * @returns Modelo configurado
 */
function createClassificationModel(
  inputSize: number,
  hiddenSize: number,
  numClasses: number
): SequentialModel {
  const model = new SequentialModel(
    "test_model",
    "Modelo de Prueba para Clasificación"
  );

  // Capa de entrada
  model.addLayer(LayerType.DENSE, {
    id: "dense_1",
    inputSize: inputSize,
    outputSize: hiddenSize,
    activation: ActivationType.RELU,
  });

  // Capa oculta
  model.addLayer(LayerType.DENSE, {
    id: "dense_2",
    inputSize: hiddenSize,
    outputSize: hiddenSize / 2,
    activation: ActivationType.RELU,
  });

  // Capa de salida con softmax para clasificación multiclase
  model.addLayer(LayerType.DENSE, {
    id: "output",
    inputSize: hiddenSize / 2,
    outputSize: numClasses,
    activation: ActivationType.SOFTMAX,
  });

  // Configurar optimizador
  model.setOptimizer(new SGDOptimizer(0.01));

  return model;
}

/**
 * Evalúa el modelo con datos de prueba y muestra métricas detalladas
 * @param model Modelo a evaluar
 * @param testInputs Datos de entrada para prueba
 * @param testTargets Datos de salida esperados
 */
function evaluateModel(
  model: SequentialModel,
  testInputs: number[][][],
  testTargets: number[][][]
): void {
  // Matriz de confusión para ver qué clases se confunden entre sí
  const numClasses = testTargets[0][0].length;
  const confusionMatrix: number[][] = Array(numClasses)
    .fill(0)
    .map(() => Array(numClasses).fill(0));

  // Evaluar cada muestra
  let correctCount = 0;

  for (let i = 0; i < testInputs.length; i++) {
    const input = testInputs[i];
    const target = testTargets[i];

    // Obtener predicción
    const output = model.predict(input);

    // Encontrar la clase predicha (índice del valor máximo)
    let predictedClass = 0;
    let maxValue = output[0][0];

    for (let j = 1; j < output[0].length; j++) {
      if (output[0][j] > maxValue) {
        maxValue = output[0][j];
        predictedClass = j;
      }
    }

    // Encontrar la clase real (índice del valor 1 en el target)
    let actualClass = 0;
    for (let j = 0; j < target[0].length; j++) {
      if (target[0][j] === 1) {
        actualClass = j;
        break;
      }
    }

    // Actualizar matriz de confusión
    confusionMatrix[actualClass][predictedClass]++;

    // Contar predicciones correctas
    if (predictedClass === actualClass) {
      correctCount++;
    }
  }

  // Calcular precisión
  const accuracy = correctCount / testInputs.length;

  // Mostrar resultados
  console.log(`\nPrecisión del modelo: ${(accuracy * 100).toFixed(2)}%`);
  console.log(
    `Predicciones correctas: ${correctCount} de ${testInputs.length}`
  );

  // Mostrar matriz de confusión
  console.log("\nMatriz de confusión:");
  console.log("Filas: clase real, Columnas: clase predicha");
  console.log(
    "     " + CLASSES.map((c) => c.substring(0, 5).padEnd(7)).join("")
  );

  for (let i = 0; i < numClasses; i++) {
    let row = CLASSES[i].substring(0, 5).padEnd(5);
    for (let j = 0; j < numClasses; j++) {
      row += `${confusionMatrix[i][j].toString().padStart(6)} `;
    }
    console.log(row);
  }

  // Mostrar métricas por clase
  console.log("\nMétricas por clase:");
  for (let i = 0; i < numClasses; i++) {
    const truePositives = confusionMatrix[i][i];
    const totalActual = confusionMatrix[i].reduce((sum, val) => sum + val, 0);
    const totalPredicted = confusionMatrix.reduce(
      (sum, row) => sum + row[i],
      0
    );

    const precision = totalPredicted > 0 ? truePositives / totalPredicted : 0;
    const recall = totalActual > 0 ? truePositives / totalActual : 0;
    const f1 =
      precision + recall > 0
        ? (2 * precision * recall) / (precision + recall)
        : 0;

    console.log(
      `${CLASSES[i].padEnd(10)}: Precisión=${(precision * 100).toFixed(
        2
      )}%, Recall=${(recall * 100).toFixed(2)}%, F1=${(f1 * 100).toFixed(2)}%`
    );
  }
}

/**
 * Función principal para validar el funcionamiento de la red neuronal
 */
async function validateNeuralNetwork(): Promise<void> {
  console.log("Iniciando validación de la red neuronal...");

  // Parámetros de configuración
  const inputSize = 1024; // Tamaño de entrada (ej. imagen 32x32 = 1024)
  const hiddenSize = 128; // Tamaño de capa oculta
  const numClasses = CLASSES.length; // Número de clases
  const numTrainSamples = 1000; // Número de muestras de entrenamiento
  const numTestSamples = 200; // Número de muestras de prueba
  const epochs = 20; // Número de épocas de entrenamiento
  const batchSize = 32; // Tamaño del lote

  // Generar datos sintéticos
  console.log("Generando datos sintéticos...");
  const { inputs: trainInputs, targets: trainTargets } = generateSyntheticData(
    numTrainSamples,
    numClasses,
    inputSize
  );
  const { inputs: testInputs, targets: testTargets } = generateSyntheticData(
    numTestSamples,
    numClasses,
    inputSize
  );

  // Crear y configurar modelo
  console.log("Creando modelo de clasificación...");
  const model = createClassificationModel(inputSize, hiddenSize, numClasses);

  // Entrenar modelo
  console.log("Entrenando modelo...");
  const startTime = Date.now();
  const trainingResults = await model.train(
    trainInputs,
    trainTargets,
    epochs,
    batchSize
  );
  const trainingTime = (Date.now() - startTime) / 1000; // en segundos

  console.log(
    `\nEntrenamiento completado en ${trainingTime.toFixed(2)} segundos`
  );
  console.log(
    `Pérdida final: ${trainingResults.loss[
      trainingResults.loss.length - 1
    ].toFixed(4)}`
  );
  console.log(
    `Precisión final: ${(
      trainingResults.accuracy[trainingResults.accuracy.length - 1] * 100
    ).toFixed(2)}%`
  );

  // Evaluar modelo
  console.log("\nEvaluando modelo con datos de prueba...");
  evaluateModel(model, testInputs, testTargets);

  // Guardar resultados
  const resultsDir = path.join(__dirname, "../../results");
  if (!fs.existsSync(resultsDir)) {
    fs.mkdirSync(resultsDir, { recursive: true });
  }

  const resultsPath = path.join(resultsDir, "validation_results.json");
  fs.writeFileSync(
    resultsPath,
    JSON.stringify(
      {
        trainingTime,
        epochs,
        finalLoss: trainingResults.loss[trainingResults.loss.length - 1],
        finalAccuracy:
          trainingResults.accuracy[trainingResults.accuracy.length - 1],
        lossHistory: trainingResults.loss,
        accuracyHistory: trainingResults.accuracy,
      },
      null,
      2
    )
  );

  console.log(`\nResultados guardados en ${resultsPath}`);
  console.log("Validación completada con éxito.");
}

// Ejecutar validación
validateNeuralNetwork().catch(console.error);
