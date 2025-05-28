import { ActivationType } from "@/core/types/ActivationType";
import { IActivation } from "@/core/interfaces/IActivation";
import { Sigmoid } from "@/neural/activations/Sigmoid";
import { ReLU } from "@/neural/activations/ReLU";
import { Tanh } from "@/neural/activations/Tanh";
import { Softmax } from "@/neural/activations/Softmax";

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
        return new Softmax();
      case ActivationType.LINEAR:
        // Para activación lineal, simplemente pasamos los valores sin transformación
        return {
          name: "Linear",
          forward: (x: number) => x,
          forwardMatrix: (x: number[][]) => x.map((row) => [...row]),
          backward: (x: number) => 1,
          backwardMatrix: (x: number[][]) => x.map((row) => row.map(() => 1)),
        };
      default:
        throw new Error(`Unknown activation type: ${type}`);
    }
  }
}
