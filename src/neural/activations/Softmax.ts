import { IActivation } from "@/core/interfaces/IActivation";

/**
 * Implementación de la función de activación Softmax
 *
 * Softmax convierte un vector de valores reales en una distribución de probabilidad,
 * donde la suma de todos los componentes es 1. Es especialmente útil para la capa
 * de salida en problemas de clasificación multiclase.
 *
 * La fórmula matemática es:
 * softmax(x_i) = exp(x_i) / sum(exp(x_j)) para todo j
 */
export class Softmax implements IActivation {
  name: string = "softmax";
  /**
   * Aplica la función Softmax a un valor escalar (no aplicable directamente)
   * Softmax debe aplicarse a un vector completo, no a valores individuales
   * @param x Valor de entrada
   * @returns Valor transformado
   */
  public forward(x: number): number {
    throw new Error(
      "Softmax cannot be applied to a single value. Use forwardVector instead."
    );
  }

  /**
   * Aplica la función Softmax a un vector
   * @param x Vector de entrada
   * @returns Vector transformado (suma = 1)
   */
  public forwardVector(x: number[]): number[] {
    // Encontrar el valor máximo para estabilidad numérica
    const maxVal = Math.max(...x);

    // Calcular exp(x_i - max) para cada elemento
    const expValues = x.map((val) => Math.exp(val - maxVal));

    // Calcular la suma de todos los valores exponenciales
    const sumExp = expValues.reduce((sum, val) => sum + val, 0);

    // Normalizar para obtener probabilidades
    return expValues.map((val) => val / sumExp);
  }

  /**
   * Aplica la función Softmax a una matriz
   * Aplica Softmax a cada fila de la matriz independientemente
   * @param x Matriz de entrada [batch_size, features]
   * @returns Matriz transformada [batch_size, features]
   */
  public forwardMatrix(x: number[][]): number[][] {
    return x.map((row) => this.forwardVector(row));
  }

  /**
   * Calcula la derivada de Softmax para un valor escalar (no aplicable directamente)
   * @param x Valor de entrada
   * @returns Derivada
   */
  public backward(x: number): number {
    throw new Error(
      "Softmax backward cannot be applied to a single value. Use backwardVector instead."
    );
  }

  /**
   * Calcula la derivada de Softmax para un vector
   * Nota: En la práctica, esta función no se usa directamente en redes neuronales
   * ya que la derivada de Softmax se combina con la derivada de Cross-Entropy
   * @param x Vector de entrada (salida de softmax)
   * @returns Matriz jacobiana (simplificada como vector)
   */
  public backwardVector(x: number[]): number[] {
    // Nota: Esta es una simplificación. La derivada completa de softmax
    // es una matriz jacobiana. En la práctica, cuando se usa con cross-entropy,
    // la derivada se simplifica a (output - target)
    return x.map((val) => val * (1 - val));
  }

  /**
   * Calcula la derivada de Softmax para una matriz
   * @param x Matriz de entrada [batch_size, features]
   * @returns Matriz de derivadas [batch_size, features]
   */
  public backwardMatrix(x: number[][]): number[][] {
    return x.map((row) => this.backwardVector(row));
  }
}
