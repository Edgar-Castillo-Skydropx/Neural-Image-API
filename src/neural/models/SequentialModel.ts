import { BaseModel } from "@/neural/models/BaseModel";
import { IOptimizer } from "@/core/interfaces/IOptimizer";
import { LayerFactory, LayerType } from "@/neural/layers/LayerFactory";
import { SGDOptimizer } from "@/neural/optimizers/SGDOptimizer";
import { CrossEntropy } from "@/neural/math/CrossEntropy";
import { ActivationType } from "@/core/types/ActivationType";

/**
 * Implementación de un modelo secuencial de red neuronal
 * Las capas se apilan secuencialmente, donde la salida de una capa es la entrada de la siguiente
 *
 * Esta implementación incluye soporte para clasificación multiclase con softmax y cross-entropy
 */
export class SequentialModel extends BaseModel {
  private useSoftmaxCrossEntropy: boolean = false;

  /**
   * Constructor del modelo secuencial
   * @param id Identificador único del modelo
   * @param name Nombre del modelo
   */
  constructor(id: string, name: string) {
    super(id, name);
  }

  /**
   * Añade una capa al modelo
   * @param layerType Tipo de capa a añadir
   * @param config Configuración de la capa
   */
  public addLayer(layerType: LayerType, config: Record<string, any>): void {
    const layer = LayerFactory.create(layerType, config);
    this.layerInstances.push(layer);
    this.layers.push(layer.id);

    // Si la última capa añadida es densa y usa softmax, activamos el modo clasificación
    if (
      layerType === LayerType.DENSE &&
      config.activation === ActivationType.SOFTMAX
    ) {
      this.useSoftmaxCrossEntropy = true;
    }
  }

  /**
   * Configura el optimizador para el entrenamiento
   * @param optimizer Optimizador a utilizar
   */
  public setOptimizer(optimizer: IOptimizer): void {
    this.optimizer = optimizer;
  }

  /**
   * Entrena el modelo con datos de entrada y salida esperada
   * @param inputs Conjunto de datos de entrada
   * @param targets Salidas esperadas correspondientes
   * @param epochs Número de épocas de entrenamiento
   * @param batchSize Tamaño del lote para entrenamiento
   */
  public async train(
    inputs: number[][][],
    targets: number[][][],
    epochs: number,
    batchSize: number
  ): Promise<Record<string, number[]>> {
    if (!this.isInitialized) {
      this.initialize();
    }

    if (!this.optimizer) {
      // Usar SGD por defecto si no se ha configurado un optimizador
      this.optimizer = new SGDOptimizer(0.01);
    }

    const accuracyHistory: number[] = [];
    const lossHistory: number[] = [];

    // Entrenamiento por épocas
    for (let epoch = 0; epoch < epochs; epoch++) {
      let totalLoss = 0;
      let correctPredictions = 0;

      // Procesar cada muestra de entrenamiento
      for (let i = 0; i < inputs.length; i++) {
        const input = inputs[i];
        const target = targets[i];

        // Propagación hacia adelante
        let output = input;
        for (const layer of this.layerInstances) {
          output = layer.forward(output);
        }

        // Calcular pérdida según el tipo de problema
        let loss: number;
        if (this.useSoftmaxCrossEntropy) {
          // Para clasificación multiclase: Cross-Entropy
          loss = CrossEntropy.loss(output, target);
        } else {
          // Para otros problemas: MSE
          loss = this.calculateMSELoss(output, target);
        }
        totalLoss += loss;

        // Verificar predicción
        if (this.isPredictionCorrect(output, target)) {
          correctPredictions++;
        }

        // Retropropagación
        let gradient: number[][];
        if (this.useSoftmaxCrossEntropy) {
          // Para clasificación multiclase: gradiente de Cross-Entropy + Softmax
          gradient = CrossEntropy.gradient(output, target);
        } else {
          // Para otros problemas: gradiente de MSE
          gradient = this.calculateMSEGradient(output, target);
        }

        // Actualizar pesos en orden inverso
        for (let j = this.layerInstances.length - 1; j >= 0; j--) {
          gradient = this.layerInstances[j].backward(
            gradient,
            this.optimizer.learningRate
          );
        }
      }

      // Calcular métricas de la época
      const accuracy = correctPredictions / inputs.length;
      const averageLoss = totalLoss / inputs.length;

      accuracyHistory.push(accuracy);
      lossHistory.push(averageLoss);

      console.log(
        `Época ${epoch + 1}/${epochs} - Pérdida: ${averageLoss.toFixed(
          4
        )} - Precisión: ${(accuracy * 100).toFixed(2)}%`
      );
    }

    return {
      accuracy: accuracyHistory,
      loss: lossHistory,
    };
  }

  /**
   * Evalúa el rendimiento del modelo
   * @param inputs Conjunto de datos de entrada para evaluación
   * @param targets Salidas esperadas correspondientes
   */
  public evaluate(
    inputs: number[][][],
    targets: number[][][]
  ): Record<string, number> {
    if (!this.isInitialized) {
      this.initialize();
    }

    let totalLoss = 0;
    let correctPredictions = 0;

    // Evaluar cada muestra
    for (let i = 0; i < inputs.length; i++) {
      const input = inputs[i];
      const target = targets[i];

      // Propagación hacia adelante
      const output = this.predict(input);

      // Calcular pérdida según el tipo de problema
      let loss: number;
      if (this.useSoftmaxCrossEntropy) {
        // Para clasificación multiclase: Cross-Entropy
        loss = CrossEntropy.loss(output, target);
      } else {
        // Para otros problemas: MSE
        loss = this.calculateMSELoss(output, target);
      }
      totalLoss += loss;

      // Verificar predicción
      if (this.isPredictionCorrect(output, target)) {
        correctPredictions++;
      }
    }

    // Calcular métricas finales
    const accuracy = correctPredictions / inputs.length;
    const averageLoss = totalLoss / inputs.length;

    return {
      accuracy,
      loss: averageLoss,
    };
  }

  /**
   * Calcula la pérdida de Error Cuadrático Medio (MSE) entre la salida y el objetivo
   * @param output Salida del modelo
   * @param target Objetivo esperado
   */
  private calculateMSELoss(output: number[][], target: number[][]): number {
    // Implementación de error cuadrático medio
    let sum = 0;
    let count = 0;

    for (let i = 0; i < output.length; i++) {
      for (let j = 0; j < output[i].length; j++) {
        const diff = output[i][j] - target[i][j];
        sum += diff * diff;
        count++;
      }
    }

    return sum / count;
  }

  /**
   * Calcula el gradiente para la retropropagación usando MSE
   * @param output Salida del modelo
   * @param target Objetivo esperado
   */
  private calculateMSEGradient(
    output: number[][],
    target: number[][]
  ): number[][] {
    // Implementación de gradiente para MSE
    const gradient: number[][] = [];

    for (let i = 0; i < output.length; i++) {
      gradient[i] = [];
      for (let j = 0; j < output[i].length; j++) {
        // Derivada de MSE: 2 * (output - target) / n
        gradient[i][j] =
          (2 * (output[i][j] - target[i][j])) /
          (output.length * output[i].length);
      }
    }

    return gradient;
  }

  /**
   * Verifica si la predicción es correcta
   * @param output Salida del modelo
   * @param target Objetivo esperado
   */
  private isPredictionCorrect(output: number[][], target: number[][]): boolean {
    // Para clasificación, comparamos el índice del valor máximo
    const getMaxIndex = (arr: number[]): number => {
      let maxIndex = 0;
      let maxValue = arr[0];

      for (let i = 1; i < arr.length; i++) {
        if (arr[i] > maxValue) {
          maxValue = arr[i];
          maxIndex = i;
        }
      }

      return maxIndex;
    };

    // Comparar para cada muestra en el batch
    for (let i = 0; i < output.length; i++) {
      const outputMaxIndex = getMaxIndex(output[i]);
      const targetMaxIndex = getMaxIndex(target[i]);

      if (outputMaxIndex !== targetMaxIndex) {
        return false;
      }
    }

    return true;
  }
}
