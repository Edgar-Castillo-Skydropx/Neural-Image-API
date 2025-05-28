import { BaseLayer } from "./BaseLayer";
import { Matrix } from "../math/Matrix";
import { ActivationType } from "../../core/types/ActivationType";

/**
 * Implementación de una capa densa (completamente conectada)
 * Realiza la operación: output = activation(input * weights + biases)
 */
export class DenseLayer extends BaseLayer {
  private weights: Matrix | null = null;
  private biases: Matrix | null = null;
  private input: Matrix | null = null;
  private output: Matrix | null = null;

  /**
   * Constructor de la capa densa
   * @param id Identificador único de la capa
   * @param inputSize Tamaño de entrada (número de neuronas)
   * @param outputSize Tamaño de salida (número de neuronas)
   * @param activationType Tipo de función de activación
   */
  constructor(
    id: string,
    inputSize: number,
    outputSize: number,
    activationType: ActivationType
  ) {
    super(id, "dense", [inputSize], [outputSize], activationType);
  }

  /**
   * Inicializa los pesos y sesgos de la capa
   * Utiliza la inicialización de Xavier/Glorot para los pesos
   */
  public initialize(): void {
    const inputSize = this.inputShape[0];
    const outputSize = this.outputShape[0];

    // Inicialización de Xavier/Glorot para mejorar la convergencia
    const stdDev = Math.sqrt(2 / (inputSize + outputSize));

    this.weights = Matrix.random(inputSize, outputSize, -stdDev, stdDev);
    this.biases = new Matrix(1, outputSize);
  }

  /**
   * Propagación hacia adelante
   * @param input Datos de entrada [batch_size, input_size]
   * @returns Datos de salida [batch_size, output_size]
   */
  public forward(input: number[][]): number[][] {
    if (!this.weights || !this.biases) {
      throw new Error("Layer not initialized");
    }

    // Convertir entrada a matriz
    this.input = Matrix.fromArray(input);

    // Calcular la salida pre-activación: input * weights + biases
    const preActivation = this.input.multiply(this.weights);

    // Añadir los sesgos a cada fila
    for (let i = 0; i < preActivation.rows; i++) {
      for (let j = 0; j < preActivation.cols; j++) {
        preActivation.data[i][j] += this.biases.data[0][j];
      }
    }

    // Aplicar la función de activación si existe
    if (this.activation) {
      this.output = Matrix.fromArray(
        this.activation.forwardMatrix(preActivation.data)
      );
    } else {
      this.output = preActivation;
    }

    return this.output.data;
  }

  /**
   * Retropropagación
   * @param outputGradient Gradiente de salida [batch_size, output_size]
   * @param learningRate Tasa de aprendizaje
   * @returns Gradiente de entrada [batch_size, input_size]
   */
  public backward(
    outputGradient: number[][],
    learningRate: number
  ): number[][] {
    if (!this.weights || !this.biases || !this.input || !this.output) {
      throw new Error("Forward pass must be called before backward pass");
    }

    // Convertir gradiente de salida a matriz
    let gradient = Matrix.fromArray(outputGradient);

    // Calcular gradiente de la activación si existe
    if (this.activation) {
      gradient = gradient.hadamardProduct(
        Matrix.fromArray(this.activation.backwardMatrix(this.output.data))
      );
    }

    // Calcular gradientes para los pesos: input^T * gradient
    const weightsGradient = this.input.transpose().multiply(gradient);

    // Calcular gradientes para los sesgos: suma de cada columna del gradiente
    const biasesGradient = new Matrix(1, this.outputShape[0]);
    for (let j = 0; j < gradient.cols; j++) {
      let sum = 0;
      for (let i = 0; i < gradient.rows; i++) {
        sum += gradient.data[i][j];
      }
      biasesGradient.data[0][j] = sum;
    }

    // Actualizar pesos y sesgos
    this.weights = this.weights.subtract(
      weightsGradient.multiplyScalar(learningRate)
    );
    this.biases = this.biases.subtract(
      biasesGradient.multiplyScalar(learningRate)
    );

    // Calcular gradiente de entrada para la capa anterior: gradient * weights^T
    const inputGradient = gradient.multiply(this.weights.transpose());

    return inputGradient.data;
  }

  /**
   * Obtiene los pesos de la capa
   */
  public getWeights(): Record<string, number[][]> {
    if (!this.weights || !this.biases) {
      throw new Error("Layer not initialized");
    }

    return {
      weights: this.weights.data,
      biases: this.biases.data,
    };
  }

  /**
   * Establece los pesos de la capa
   * @param weights Pesos a establecer
   */
  public setWeights(weights: Record<string, number[][]>): void {
    if (!weights.weights || !weights.biases) {
      throw new Error("Invalid weights object");
    }

    this.weights = Matrix.fromArray(weights.weights);
    this.biases = Matrix.fromArray(weights.biases);
  }

  /**
   * Carga la configuración de la capa desde formato JSON
   * @param config Configuración en formato JSON
   */
  public fromJSON(config: Record<string, any>): void {
    if (!config.weights || !config.biases) {
      throw new Error("Invalid layer configuration");
    }

    this.setWeights({
      weights: config.weights,
      biases: config.biases,
    });
  }
}
