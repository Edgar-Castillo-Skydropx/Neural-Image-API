import { BaseActivation } from './BaseActivation';

/**
 * Implementación de la función de activación ReLU (Rectified Linear Unit)
 * f(x) = max(0, x)
 */
export class ReLU extends BaseActivation {
  /**
   * Constructor de la función ReLU
   */
  constructor() {
    super('relu');
  }

  /**
   * Implementación de la función ReLU
   * @param input Valor de entrada
   * @returns Valor activado mediante ReLU
   */
  public forward(input: number): number {
    return Math.max(0, input);
  }

  /**
   * Derivada de la función ReLU
   * f'(x) = 1 si x > 0, 0 en caso contrario
   * @param input Valor de entrada
   * @returns Derivada de ReLU en el punto dado
   */
  public backward(input: number): number {
    return input > 0 ? 1 : 0;
  }
}
