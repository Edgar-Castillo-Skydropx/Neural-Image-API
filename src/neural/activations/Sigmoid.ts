import { BaseActivation } from "@/neural/activations/BaseActivation";

/**
 * Implementación de la función de activación Sigmoid
 * σ(x) = 1 / (1 + e^(-x))
 */
export class Sigmoid extends BaseActivation {
  /**
   * Constructor de la función Sigmoid
   */
  constructor() {
    super("sigmoid");
  }

  /**
   * Implementación de la función Sigmoid
   * @param input Valor de entrada
   * @returns Valor activado mediante Sigmoid
   */
  public forward(input: number): number {
    return 1 / (1 + Math.exp(-input));
  }

  /**
   * Derivada de la función Sigmoid
   * σ'(x) = σ(x) * (1 - σ(x))
   * @param input Valor de entrada
   * @returns Derivada de Sigmoid en el punto dado
   */
  public backward(input: number): number {
    const sigmoid = this.forward(input);
    return sigmoid * (1 - sigmoid);
  }
}
