import { ILayer } from "@/core/interfaces/ILayer";
import { InputLayer } from "@/neural/layers/InputLayer";
import { DenseLayer } from "@/neural/layers/DenseLayer";
import { ConvolutionalLayer } from "@/neural/layers/ConvolutionalLayer";
import { ActivationType } from "@/core/types/ActivationType";

/**
 * Tipos de capas disponibles para la red neuronal
 */
export enum LayerType {
  INPUT = "input",
  DENSE = "dense",
  CONVOLUTIONAL = "convolutional",
  POOLING = "pooling",
}

/**
 * Factory para crear capas neuronales
 * Implementa el patrón Factory para instanciar diferentes tipos de capas
 */
export class LayerFactory {
  /**
   * Crea una instancia de capa según el tipo y configuración especificados
   * @param type Tipo de capa a crear
   * @param config Configuración de la capa
   * @returns Instancia de la capa
   */
  public static create(type: LayerType, config: Record<string, any>): ILayer {
    const id = config.id || `layer_${Date.now()}`;

    switch (type) {
      case LayerType.INPUT:
        return new InputLayer(id, config.inputShape);

      case LayerType.DENSE:
        return new DenseLayer(
          id,
          config.inputSize,
          config.outputSize,
          config.activation || ActivationType.SIGMOID
        );

      case LayerType.CONVOLUTIONAL:
        return new ConvolutionalLayer(
          id,
          config.inputShape,
          config.kernelSize,
          config.filters,
          config.stride || 1,
          config.padding || 0,
          config.activation || ActivationType.RELU
        );

      case LayerType.POOLING:
        throw new Error("Pooling layer not implemented yet");

      default:
        throw new Error(`Unknown layer type: ${type}`);
    }
  }
}
