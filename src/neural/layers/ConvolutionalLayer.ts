import { BaseLayer } from './BaseLayer';
import { ActivationType } from '../../core/types/ActivationType';
import { Matrix } from '../math/Matrix';

/**
 * Implementación de una capa convolucional para procesamiento de imágenes
 * Realiza la operación de convolución 2D sobre los datos de entrada
 */
export class ConvolutionalLayer extends BaseLayer {
  private filters: Matrix[] = [];
  private biases: number[] = [];
  private kernelSize: number;
  private stride: number;
  private padding: number;
  private inputChannels: number;
  private outputChannels: number;
  
  private input: number[][][] | null = null;
  private output: number[][][] | null = null;

  /**
   * Constructor de la capa convolucional
   * @param id Identificador único de la capa
   * @param inputShape Forma de entrada [altura, anchura, canales]
   * @param kernelSize Tamaño del kernel (filtro)
   * @param filters Número de filtros (canales de salida)
   * @param stride Paso de la convolución
   * @param padding Relleno alrededor de la entrada
   * @param activationType Tipo de función de activación
   */
  constructor(
    id: string,
    inputShape: number[],
    kernelSize: number,
    filters: number,
    stride: number = 1,
    padding: number = 0,
    activationType: ActivationType
  ) {
    // Calcular la forma de salida
    const inputHeight = inputShape[0];
    const inputWidth = inputShape[1];
    const inputChannels = inputShape[2] || 1;
    
    const outputHeight = Math.floor((inputHeight + 2 * padding - kernelSize) / stride) + 1;
    const outputWidth = Math.floor((inputWidth + 2 * padding - kernelSize) / stride) + 1;
    
    super(
      id,
      'convolutional',
      inputShape,
      [outputHeight, outputWidth, filters],
      activationType
    );
    
    this.kernelSize = kernelSize;
    this.stride = stride;
    this.padding = padding;
    this.inputChannels = inputChannels;
    this.outputChannels = filters;
  }

  /**
   * Inicializa los filtros y sesgos de la capa
   * Utiliza la inicialización de He para los pesos
   */
  public initialize(): void {
    // Inicialización de He para redes con ReLU
    const stdDev = Math.sqrt(2 / (this.kernelSize * this.kernelSize * this.inputChannels));
    
    // Inicializar filtros
    for (let i = 0; i < this.outputChannels; i++) {
      const filter = Matrix.random(
        this.kernelSize * this.kernelSize * this.inputChannels,
        1,
        -stdDev,
        stdDev
      );
      this.filters.push(filter);
      this.biases.push(0); // Inicializar sesgos a cero
    }
  }

  /**
   * Aplica padding a la imagen de entrada
   * @param input Imagen de entrada [altura, anchura, canales]
   * @returns Imagen con padding [altura+2*padding, anchura+2*padding, canales]
   */
  private applyPadding(input: number[][][]): number[][][] {
    if (this.padding === 0) {
      return input;
    }
    
    const height = input.length;
    const width = input[0].length;
    const channels = input[0][0].length;
    
    const paddedHeight = height + 2 * this.padding;
    const paddedWidth = width + 2 * this.padding;
    
    // Crear matriz con padding inicializada a cero
    const padded: number[][][] = Array(paddedHeight)
      .fill(0)
      .map(() => Array(paddedWidth)
        .fill(0)
        .map(() => Array(channels).fill(0))
      );
    
    // Copiar los valores originales
    for (let i = 0; i < height; i++) {
      for (let j = 0; j < width; j++) {
        for (let c = 0; c < channels; c++) {
          padded[i + this.padding][j + this.padding][c] = input[i][j][c];
        }
      }
    }
    
    return padded;
  }

  /**
   * Extrae un parche de la imagen de entrada
   * @param input Imagen de entrada
   * @param row Fila inicial
   * @param col Columna inicial
   * @returns Parche aplanado como vector
   */
  private extractPatch(input: number[][][], row: number, col: number): number[] {
    const patch: number[] = [];
    
    for (let c = 0; c < this.inputChannels; c++) {
      for (let i = 0; i < this.kernelSize; i++) {
        for (let j = 0; j < this.kernelSize; j++) {
          patch.push(input[row + i][col + j][c]);
        }
      }
    }
    
    return patch;
  }

  /**
   * Propagación hacia adelante
   * @param input Datos de entrada [batch_size, input_height * input_width * input_channels]
   * @returns Datos de salida [batch_size, output_height * output_width * output_channels]
   */
  public forward(input: number[][]): number[][] {
    const batchSize = input.length;
    const inputHeight = this.inputShape[0];
    const inputWidth = this.inputShape[1];
    
    const outputHeight = this.outputShape[0];
    const outputWidth = this.outputShape[1];
    
    // Convertir entrada plana a formato de imagen [batch_size, height, width, channels]
    const inputImages: number[][][][] = [];
    for (let b = 0; b < batchSize; b++) {
      const image: number[][][] = Array(inputHeight)
        .fill(0)
        .map(() => Array(inputWidth)
          .fill(0)
          .map(() => Array(this.inputChannels).fill(0))
        );
      
      let idx = 0;
      for (let i = 0; i < inputHeight; i++) {
        for (let j = 0; j < inputWidth; j++) {
          for (let c = 0; c < this.inputChannels; c++) {
            image[i][j][c] = input[b][idx++];
          }
        }
      }
      
      inputImages.push(image);
    }
    
    // Guardar entrada para backward pass
    this.input = inputImages.map(img => this.applyPadding(img));
    
    // Calcular salida para cada imagen en el batch
    const outputBatch: number[][] = [];
    
    for (let b = 0; b < batchSize; b++) {
      const paddedInput = this.input[b];
      const outputImage: number[][][] = Array(outputHeight)
        .fill(0)
        .map(() => Array(outputWidth)
          .fill(0)
          .map(() => Array(this.outputChannels).fill(0))
        );
      
      // Aplicar convolución
      for (let i = 0; i <= paddedInput.length - this.kernelSize; i += this.stride) {
        for (let j = 0; j <= paddedInput[0].length - this.kernelSize; j += this.stride) {
          const patch = this.extractPatch(paddedInput, i, j);
          
          for (let f = 0; f < this.outputChannels; f++) {
            let sum = this.biases[f];
            
            // Producto escalar del parche con el filtro
            for (let p = 0; p < patch.length; p++) {
              sum += patch[p] * this.filters[f].data[p][0];
            }
            
            // Aplicar activación
            if (this.activation) {
              sum = this.activation.forward(sum);
            }
            
            const outI = Math.floor(i / this.stride);
            const outJ = Math.floor(j / this.stride);
            outputImage[outI][outJ][f] = sum;
          }
        }
      }
      
      // Aplanar la salida
      const flatOutput: number[] = [];
      for (let i = 0; i < outputHeight; i++) {
        for (let j = 0; j < outputWidth; j++) {
          for (let f = 0; f < this.outputChannels; f++) {
            flatOutput.push(outputImage[i][j][f]);
          }
        }
      }
      
      outputBatch.push(flatOutput);
    }
    
    // Guardar salida para backward pass
    this.output = outputBatch.map(flat => {
      const img: number[][][] = Array(outputHeight)
        .fill(0)
        .map(() => Array(outputWidth)
          .fill(0)
          .map(() => Array(this.outputChannels).fill(0))
        );
      
      let idx = 0;
      for (let i = 0; i < outputHeight; i++) {
        for (let j = 0; j < outputWidth; j++) {
          for (let f = 0; f < this.outputChannels; f++) {
            img[i][j][f] = flat[idx++];
          }
        }
      }
      
      return img;
    });
    
    return outputBatch;
  }

  /**
   * Retropropagación
   * @param outputGradient Gradiente de salida [batch_size, output_height * output_width * output_channels]
   * @param learningRate Tasa de aprendizaje
   * @returns Gradiente de entrada [batch_size, input_height * input_width * input_channels]
   */
  public backward(outputGradient: number[][], learningRate: number): number[][] {
    if (!this.input || !this.output) {
      throw new Error('Forward pass must be called before backward pass');
    }
    
    const batchSize = outputGradient.length;
    const inputHeight = this.inputShape[0];
    const inputWidth = this.inputShape[1];
    const outputHeight = this.outputShape[0];
    const outputWidth = this.outputShape[1];
    
    // Convertir gradiente plano a formato de imagen [batch_size, height, width, channels]
    const outputGradImages: number[][][][] = [];
    for (let b = 0; b < batchSize; b++) {
      const gradImage: number[][][] = Array(outputHeight)
        .fill(0)
        .map(() => Array(outputWidth)
          .fill(0)
          .map(() => Array(this.outputChannels).fill(0))
        );
      
      let idx = 0;
      for (let i = 0; i < outputHeight; i++) {
        for (let j = 0; j < outputWidth; j++) {
          for (let f = 0; f < this.outputChannels; f++) {
            gradImage[i][j][f] = outputGradient[b][idx++];
          }
        }
      }
      
      outputGradImages.push(gradImage);
    }
    
    // Inicializar gradientes de filtros y sesgos
    const filterGradients: Matrix[] = [];
    const biasGradients: number[] = Array(this.outputChannels).fill(0);
    
    for (let f = 0; f < this.outputChannels; f++) {
      filterGradients.push(new Matrix(this.kernelSize * this.kernelSize * this.inputChannels, 1));
    }
    
    // Inicializar gradientes de entrada
    const inputGradients: number[][][] = Array(batchSize)
      .fill(0)
      .map(() => Array(inputHeight)
        .fill(0)
        .map(() => Array(inputWidth)
          .fill(0)
          .map(() => Array(this.inputChannels).fill(0))
        )
      );
    
    // Calcular gradientes
    for (let b = 0; b < batchSize; b++) {
      const paddedInput = this.input[b];
      const outputGrad = outputGradImages[b];
      
      // Calcular gradientes de filtros y sesgos
      for (let i = 0; i < outputHeight; i++) {
        for (let j = 0; j < outputWidth; j++) {
          const inputI = i * this.stride;
          const inputJ = j * this.stride;
          const patch = this.extractPatch(paddedInput, inputI, inputJ);
          
          for (let f = 0; f < this.outputChannels; f++) {
            const gradValue = outputGrad[i][j][f];
            
            // Gradiente del sesgo
            biasGradients[f] += gradValue;
            
            // Gradiente del filtro
            for (let p = 0; p < patch.length; p++) {
              filterGradients[f].data[p][0] += patch[p] * gradValue;
            }
          }
        }
      }
    }
    
    // Actualizar filtros y sesgos
    for (let f = 0; f < this.outputChannels; f++) {
      // Actualizar filtro
      this.filters[f] = this.filters[f].subtract(
        filterGradients[f].multiplyScalar(learningRate / batchSize)
      );
      
      // Actualizar sesgo
      this.biases[f] -= learningRate * biasGradients[f] / batchSize;
    }
    
    // Calcular gradiente de entrada (simplificado - en una implementación real sería más complejo)
    const inputGradientBatch: number[][] = [];
    for (let b = 0; b < batchSize; b++) {
      const flatGrad: number[] = [];
      for (let i = 0; i < inputHeight; i++) {
        for (let j = 0; j < inputWidth; j++) {
          for (let c = 0; c < this.inputChannels; c++) {
            flatGrad.push(0); // Simplificado
          }
        }
      }
      inputGradientBatch.push(flatGrad);
    }
    
    return inputGradientBatch;
  }

  /**
   * Obtiene los pesos de la capa
   */
  public getWeights(): Record<string, number[][]> {
    const weights: Record<string, number[][]> = {
      biases: [this.biases]
    };
    
    for (let f = 0; f < this.filters.length; f++) {
      weights[`filter_${f}`] = this.filters[f].data;
    }
    
    return weights;
  }

  /**
   * Establece los pesos de la capa
   * @param weights Pesos a establecer
   */
  public setWeights(weights: Record<string, number[][]>): void {
    if (!weights.biases) {
      throw new Error('Invalid weights object: missing biases');
    }
    
    this.biases = weights.biases[0];
    
    for (let f = 0; f < this.outputChannels; f++) {
      const filterKey = `filter_${f}`;
      if (!weights[filterKey]) {
        throw new Error(`Invalid weights object: missing ${filterKey}`);
      }
      
      this.filters[f] = Matrix.fromArray(weights[filterKey]);
    }
  }

  /**
   * Carga la configuración de la capa desde formato JSON
   * @param config Configuración en formato JSON
   */
  public fromJSON(config: Record<string, any>): void {
    if (!config.weights) {
      throw new Error('Invalid layer configuration: missing weights');
    }
    
    this.setWeights(config.weights);
  }
}
