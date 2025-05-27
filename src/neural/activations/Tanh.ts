import { BaseActivation } from "@/neural/activations/BaseActivation";

/**
 * Implementación de la función de activación Tanh (Tangente Hiperbólica)
 * tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
 */
export class Tanh extends BaseActivation {
  /**
   * Constructor de la función Tanh
   */
  constructor() {
    super("tanh");
  }

  /**
   * Implementación de la función Tanh
   * @param input Valor de entrada
   * @returns Valor activado mediante Tanh
   */
  public forward(input: number): number {
    return Math.tanh(input);
  }

  /**
   * Derivada de la función Tanh
   * tanh'(x) = 1 - tanh²(x)
   * @param input Valor de entrada
   * @returns Derivada de Tanh en el punto dado
   */
  public backward(input: number): number {
    const tanhValue = this.forward(input);
    return 1 - tanhValue * tanhValue;
  }
}
