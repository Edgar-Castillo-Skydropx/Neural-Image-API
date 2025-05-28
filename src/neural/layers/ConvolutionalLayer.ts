import { BaseLayer } from "./BaseLayer";
import { Matrix } from "@/neural/math/Matrix";
import { ActivationType } from "@/core/types/ActivationType";

/**
 * Interfaz para la configuración de una capa convolucional
 */
export interface ConvolutionalLayerConfig {
  inputShape: [number, number, number]; // [altura, anchura, canales]
  filters: number; // Número de filtros
  kernelSize: [number, number]; // Tamaño del kernel [altura, anchura]
  strides?: [number, number]; // Pasos [vertical, horizontal]
  padding?: "valid" | "same"; // Tipo de padding
  activation?: ActivationType; // Tipo de activación
}

/**
 * Implementación de una capa convolucional para redes neuronales
 * Realiza operaciones de convolución 2D sobre datos de entrada
 */
export class ConvolutionalLayer extends BaseLayer {
  private filters: number;
  private kernelSize: [number, number];
  private strides: [number, number];
  private padding: "valid" | "same";
  private kernels: Matrix[];
  private biases: Matrix;

  // Almacenamiento para propagación hacia atrás
  private lastInput: number[][] | null = null;
  private lastOutput: number[][] | null = null;
  private inputVolumes: number[][][] | null = null;
  private outputHeight: number = 0;
  private outputWidth: number = 0;

  /**
   * Constructor de la capa convolucional
   * @param id Identificador único de la capa
   * @param config Configuración de la capa convolucional
   */
  constructor(id: string, config: ConvolutionalLayerConfig) {
    // Calcular dimensiones de salida para pasar a BaseLayer
    const inputShape = config.inputShape;
    const kernelSize = config.kernelSize;
    const strides = config.strides || [1, 1];
    const padding = config.padding || "valid";

    let outputHeight: number;
    let outputWidth: number;

    if (padding === "valid") {
      outputHeight =
        Math.floor((inputShape[0] - kernelSize[0]) / strides[0]) + 1;
      outputWidth =
        Math.floor((inputShape[1] - kernelSize[1]) / strides[1]) + 1;
    } else {
      // 'same'
      outputHeight = Math.ceil(inputShape[0] / strides[0]);
      outputWidth = Math.ceil(inputShape[1] / strides[1]);
    }

    // Crear forma de salida: [altura, anchura, filtros]
    const outputShape: number[] = [outputHeight, outputWidth, config.filters];

    // Llamar al constructor de BaseLayer con los parámetros correctos
    super(id, "convolutional", inputShape, outputShape, config.activation);

    // Guardar configuración específica de la capa convolucional
    this.filters = config.filters;
    this.kernelSize = kernelSize;
    this.strides = strides;
    this.padding = padding;
    this.outputHeight = outputHeight;
    this.outputWidth = outputWidth;
    this.kernels = [];
    this.biases = new Matrix(1, 1);

    // Inicializar kernels y biases
    this.initialize();
  }

  /**
   * Inicializa los pesos de la capa convolucional
   */
  public initialize(): void {
    const inputChannels = this.inputShape[2];

    // Inicializar kernels (filtros)
    this.kernels = [];
    for (let i = 0; i < this.filters; i++) {
      // Crear un kernel para cada filtro
      // Cada kernel tiene dimensiones [kernelHeight, kernelWidth, inputChannels]
      const kernel = new Matrix(
        this.kernelSize[0],
        this.kernelSize[1] * inputChannels
      );

      // Inicialización Xavier/Glorot para mejorar convergencia
      const stdDev = Math.sqrt(
        2.0 /
          (this.kernelSize[0] * this.kernelSize[1] * inputChannels +
            this.filters)
      );
      kernel.randomize(-stdDev, stdDev);

      this.kernels.push(kernel);
    }

    // Inicializar biases (uno por filtro)
    this.biases = new Matrix(1, this.filters);
    this.biases.randomize(-0.1, 0.1);
  }

  /**
   * Aplica padding a un volumen de entrada
   * @param input Volumen de entrada [altura, anchura, canales]
   * @param padHeight Cantidad de padding vertical
   * @param padWidth Cantidad de padding horizontal
   * @returns Volumen con padding aplicado
   */
  private applyPadding(
    input: number[][][],
    padHeight: number,
    padWidth: number
  ): number[][][] {
    if (padHeight === 0 && padWidth === 0) {
      return input;
    }

    const [height, width, channels] = [
      input.length,
      input[0].length,
      input[0][0].length,
    ];
    const paddedHeight = height + 2 * padHeight;
    const paddedWidth = width + 2 * padWidth;

    // Crear volumen con padding (inicializado a ceros)
    const padded: number[][][] = Array(paddedHeight)
      .fill(0)
      .map(() =>
        Array(paddedWidth)
          .fill(0)
          .map(() => Array(channels).fill(0))
      );

    // Copiar valores originales al centro del volumen con padding
    for (let h = 0; h < height; h++) {
      for (let w = 0; w < width; w++) {
        for (let c = 0; c < channels; c++) {
          padded[h + padHeight][w + padWidth][c] = input[h][w][c];
        }
      }
    }

    return padded;
  }

  /**
   * Convierte una matriz 2D a un volumen 3D
   * @param matrix Matriz 2D [filas, columnas]
   * @param height Altura del volumen
   * @param width Anchura del volumen
   * @param channels Canales del volumen
   * @returns Volumen 3D [altura, anchura, canales]
   */
  private matrixToVolume(
    matrix: number[][],
    height: number,
    width: number,
    channels: number
  ): number[][][] {
    const volume: number[][][] = Array(height)
      .fill(0)
      .map(() =>
        Array(width)
          .fill(0)
          .map(() => Array(channels).fill(0))
      );

    let idx = 0;
    for (let h = 0; h < height; h++) {
      for (let w = 0; w < width; w++) {
        for (let c = 0; c < channels; c++) {
          volume[h][w][c] = matrix[0][idx++];
        }
      }
    }

    return volume;
  }

  /**
   * Convierte un volumen 3D a una matriz 2D
   * @param volume Volumen 3D [altura, anchura, canales]
   * @returns Matriz 2D [1, altura*anchura*canales]
   */
  private volumeToMatrix(volume: number[][][]): number[][] {
    const height = volume.length;
    const width = volume[0].length;
    const channels = volume[0][0].length;

    const matrix: number[][] = [[]];

    for (let h = 0; h < height; h++) {
      for (let w = 0; w < width; w++) {
        for (let c = 0; c < channels; c++) {
          matrix[0].push(volume[h][w][c]);
        }
      }
    }

    return matrix;
  }

  /**
   * Realiza la operación de convolución en un volumen de entrada
   * @param input Volumen de entrada [altura, anchura, canales]
   * @returns Volumen de salida [altura, anchura, filtros]
   */
  private convolve(input: number[][][]): number[][][] {
    const inputHeight = input.length;
    const inputWidth = input[0].length;
    const inputChannels = input[0][0].length;

    // Inicializar volumen de salida
    const output: number[][][] = Array(this.outputHeight)
      .fill(0)
      .map(() =>
        Array(this.outputWidth)
          .fill(0)
          .map(() => Array(this.filters).fill(0))
      );

    // Para cada filtro
    for (let f = 0; f < this.filters; f++) {
      // Para cada posición de salida
      for (let y = 0; y < this.outputHeight; y++) {
        for (let x = 0; x < this.outputWidth; x++) {
          // Calcular posición inicial en el input
          const startY = y * this.strides[0];
          const startX = x * this.strides[1];

          let sum = 0;

          // Para cada posición del kernel
          for (let ky = 0; ky < this.kernelSize[0]; ky++) {
            for (let kx = 0; kx < this.kernelSize[1]; kx++) {
              // Para cada canal de entrada
              for (let c = 0; c < inputChannels; c++) {
                const inputY = startY + ky;
                const inputX = startX + kx;

                // Verificar límites
                if (
                  inputY >= 0 &&
                  inputY < inputHeight &&
                  inputX >= 0 &&
                  inputX < inputWidth
                ) {
                  // Obtener valor del kernel y del input
                  const kernelValue = this.kernels[f].get(
                    ky,
                    kx * inputChannels + c
                  );
                  const inputValue = input[inputY][inputX][c];

                  // Acumular producto
                  sum += kernelValue * inputValue;
                }
              }
            }
          }

          // Añadir bias y guardar en salida
          output[y][x][f] = sum + this.biases.get(0, f);
        }
      }
    }

    return output;
  }

  /**
   * Propagación hacia adelante
   * @param input Matriz de entrada [batch, features]
   * @returns Matriz de salida [batch, features]
   */
  public forward(input: number[][]): number[][] {
    // Guardar entrada para backpropagation
    this.lastInput = input;

    // Convertir matriz 2D a volumen 3D
    const batchSize = input.length;
    const inputVolumes: number[][][] = [];

    for (let b = 0; b < batchSize; b++) {
      // Convertir fila de la matriz a volumen
      const volume = this.matrixToVolume(
        [input[b]],
        this.inputShape[0],
        this.inputShape[1],
        this.inputShape[2]
      );
      inputVolumes.push(volume);
    }

    // Guardar volúmenes de entrada para backpropagation
    this.inputVolumes = inputVolumes;

    // Procesar cada volumen en el batch
    const outputMatrix: number[][] = [];

    for (let b = 0; b < batchSize; b++) {
      // Calcular padding si es necesario
      let paddedInput = inputVolumes[b];

      if (this.padding === "same") {
        const padHeight = Math.floor((this.kernelSize[0] - 1) / 2);
        const padWidth = Math.floor((this.kernelSize[1] - 1) / 2);
        paddedInput = this.applyPadding(paddedInput, padHeight, padWidth);
      }

      // Aplicar convolución
      const convOutput = this.convolve(paddedInput);

      // Aplicar activación si existe
      let activatedOutput: number[][][];

      if (this.activation) {
        // Convertir volumen a matriz para aplicar activación
        const flatConvOutput = this.volumeToMatrix(convOutput);
        const activatedFlatOutput =
          this.activation.forwardMatrix(flatConvOutput);

        // Convertir de vuelta a volumen
        activatedOutput = this.matrixToVolume(
          activatedFlatOutput,
          this.outputHeight,
          this.outputWidth,
          this.filters
        );
      } else {
        activatedOutput = convOutput;
      }

      // Convertir volumen 3D a fila de matriz 2D
      const outputRow = this.volumeToMatrix(activatedOutput)[0];
      outputMatrix.push(outputRow);
    }

    // Guardar salida para backpropagation
    this.lastOutput = outputMatrix;

    return outputMatrix;
  }

  /**
   * Propagación hacia atrás
   * @param gradient Gradiente de la capa siguiente
   * @param learningRate Tasa de aprendizaje
   * @returns Gradiente para la capa anterior
   */
  public backward(gradient: number[][], learningRate: number): number[][] {
    if (!this.lastInput || !this.lastOutput || !this.inputVolumes) {
      throw new Error(
        "No se puede realizar backpropagation sin forward previo"
      );
    }

    const batchSize = gradient.length;

    // Aplicar derivada de la activación si existe
    let adjustedGradient = gradient;
    if (this.activation) {
      adjustedGradient = this.activation.backwardMatrix(this.lastOutput);

      // Multiplicar elemento a elemento
      for (let i = 0; i < gradient.length; i++) {
        for (let j = 0; j < gradient[i].length; j++) {
          adjustedGradient[i][j] *= gradient[i][j];
        }
      }
    }

    // Convertir gradiente a volúmenes 3D
    const gradientVolumes: number[][][] = [];

    for (let b = 0; b < batchSize; b++) {
      const volume = this.matrixToVolume(
        [adjustedGradient[b]],
        this.outputHeight,
        this.outputWidth,
        this.filters
      );
      gradientVolumes.push(volume);
    }

    // Inicializar gradientes para kernels y biases
    const kernelGradients: Matrix[] = [];
    for (let f = 0; f < this.filters; f++) {
      kernelGradients.push(
        new Matrix(this.kernelSize[0], this.kernelSize[1] * this.inputShape[2])
      );
    }

    const biasGradients = new Matrix(1, this.filters);

    // Inicializar gradiente para la capa anterior
    const inputGradient: number[][] = [];
    for (let b = 0; b < batchSize; b++) {
      inputGradient.push(Array(this.lastInput[b].length).fill(0));
    }

    // Para cada ejemplo en el batch
    for (let b = 0; b < batchSize; b++) {
      // Obtener volumen de entrada
      const inputVolume = this.inputVolumes[b];

      // Calcular padding si es necesario
      let paddedInput = inputVolume;

      if (this.padding === "same") {
        const padHeight = Math.floor((this.kernelSize[0] - 1) / 2);
        const padWidth = Math.floor((this.kernelSize[1] - 1) / 2);
        paddedInput = this.applyPadding(paddedInput, padHeight, padWidth);
      }

      // Obtener gradiente para este ejemplo
      const gradientVolume = gradientVolumes[b];

      // Inicializar gradiente para la entrada de este ejemplo
      const inputGradientVolume: number[][][] = Array(this.inputShape[0])
        .fill(0)
        .map(() =>
          Array(this.inputShape[1])
            .fill(0)
            .map(() => Array(this.inputShape[2]).fill(0))
        );

      // Para cada filtro
      for (let f = 0; f < this.filters; f++) {
        // Para cada posición de salida
        for (let y = 0; y < this.outputHeight; y++) {
          for (let x = 0; x < this.outputWidth; x++) {
            // Obtener gradiente en esta posición
            const gradValue = gradientVolume[y][x][f];

            // Actualizar bias gradient
            biasGradients.add(0, f, gradValue);

            // Calcular posición inicial en el input
            const startY = y * this.strides[0];
            const startX = x * this.strides[1];

            // Para cada posición del kernel
            for (let ky = 0; ky < this.kernelSize[0]; ky++) {
              for (let kx = 0; kx < this.kernelSize[1]; kx++) {
                // Para cada canal de entrada
                for (let c = 0; c < this.inputShape[2]; c++) {
                  const inputY = startY + ky;
                  const inputX = startX + kx;

                  // Verificar límites
                  if (
                    inputY >= 0 &&
                    inputY < paddedInput.length &&
                    inputX >= 0 &&
                    inputX < paddedInput[0].length
                  ) {
                    // Obtener valor de entrada
                    const inputValue = paddedInput[inputY][inputX][c];

                    // Actualizar gradiente del kernel
                    kernelGradients[f].add(
                      ky,
                      kx * this.inputShape[2] + c,
                      gradValue * inputValue
                    );

                    // Si estamos dentro de los límites del input original (sin padding)
                    if (
                      this.padding === "valid" ||
                      (inputY >= 0 &&
                        inputY < this.inputShape[0] &&
                        inputX >= 0 &&
                        inputX < this.inputShape[1])
                    ) {
                      // Obtener valor del kernel
                      const kernelValue = this.kernels[f].get(
                        ky,
                        kx * this.inputShape[2] + c
                      );

                      // Actualizar gradiente de entrada
                      // Ajustar coordenadas si hay padding
                      const adjustedY =
                        this.padding === "same"
                          ? inputY - Math.floor((this.kernelSize[0] - 1) / 2)
                          : inputY;
                      const adjustedX =
                        this.padding === "same"
                          ? inputX - Math.floor((this.kernelSize[1] - 1) / 2)
                          : inputX;

                      if (
                        adjustedY >= 0 &&
                        adjustedY < this.inputShape[0] &&
                        adjustedX >= 0 &&
                        adjustedX < this.inputShape[1]
                      ) {
                        inputGradientVolume[adjustedY][adjustedX][c] +=
                          gradValue * kernelValue;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }

      // Convertir volumen de gradiente a fila de matriz
      const inputGradientRow = this.volumeToMatrix(inputGradientVolume)[0];

      // Actualizar gradiente de entrada para este ejemplo
      for (let i = 0; i < inputGradientRow.length; i++) {
        inputGradient[b][i] = inputGradientRow[i];
      }
    }

    // Actualizar kernels y biases con los gradientes
    for (let f = 0; f < this.filters; f++) {
      // Actualizar kernel
      for (let i = 0; i < this.kernelSize[0]; i++) {
        for (let j = 0; j < this.kernelSize[1] * this.inputShape[2]; j++) {
          const gradValue = kernelGradients[f].get(i, j) / batchSize;
          const currentValue = this.kernels[f].get(i, j);
          this.kernels[f].set(i, j, currentValue - learningRate * gradValue);
        }
      }

      // Actualizar bias
      const biasGradValue = biasGradients.get(0, f) / batchSize;
      const currentBiasValue = this.biases.get(0, f);
      this.biases.set(0, f, currentBiasValue - learningRate * biasGradValue);
    }

    return inputGradient;
  }

  /**
   * Obtiene los pesos de la capa
   */
  public getWeights(): Record<string, number[][]> {
    const weights: Record<string, number[][]> = {
      biases: this.biases.data,
    };

    // Añadir kernels
    for (let i = 0; i < this.filters; i++) {
      weights[`kernel_${i}`] = this.kernels[i].data;
    }

    return weights;
  }

  /**
   * Establece los pesos de la capa
   * @param weights Pesos a establecer
   */
  public setWeights(weights: Record<string, number[][]>): void {
    if (weights.biases) {
      this.biases = Matrix.fromArray(weights.biases);
    }

    // Cargar kernels
    for (let i = 0; i < this.filters; i++) {
      const kernelKey = `kernel_${i}`;
      if (weights[kernelKey]) {
        this.kernels[i] = Matrix.fromArray(weights[kernelKey]);
      }
    }
  }

  /**
   * Carga la configuración de la capa desde formato JSON
   * @param config Configuración en formato JSON
   */
  public fromJSON(config: Record<string, any>): void {
    if (config.weights) {
      this.setWeights(config.weights);
    }
  }
}
