import { ActivationType } from "@/core/types/ActivationType";
import { IActivation } from "@/core/interfaces/IActivation";
import { Sigmoid } from "@/neural/activations/Sigmoid";
import { ReLU } from "@/neural/activations/ReLU";
import { Tanh } from "@/neural/activations/Tanh";

/**
 * Factory para crear funciones de activación
 * Implementa el patrón Factory para instanciar diferentes tipos de activaciones
 */
export class ActivationFactory {
  /**
   * Crea una instancia de función de activación según el tipo especificado
   * @param type Tipo de activación a crear
   * @returns Instancia de la función de activación
   */
  public static create(type: ActivationType): IActivation {
    switch (type) {
      case ActivationType.SIGMOID:
        return new Sigmoid();
      case ActivationType.RELU:
        return new ReLU();
      case ActivationType.TANH:
        return new Tanh();
      case ActivationType.SOFTMAX:
        throw new Error("Softmax activation not implemented yet");
      case ActivationType.LINEAR:
        throw new Error("Linear activation not implemented yet");
      default:
        throw new Error(`Unknown activation type: ${type}`);
    }
  }
}
