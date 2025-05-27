import { IActivation } from "@/core/interfaces/IActivation";

/**
 * Clase base abstracta para todas las funciones de activación
 * Implementa la interfaz IActivation y proporciona funcionalidad común
 */
export abstract class BaseActivation implements IActivation {
  public readonly name: string;

  /**
   * Constructor de la función de activación base
   * @param name Nombre de la función de activación
   */
  constructor(name: string) {
    this.name = name;
  }

  /**
   * Función de activación directa (debe ser implementada por clases hijas)
   * @param input Valor de entrada
   */
  public abstract forward(input: number): number;

  /**
   * Derivada de la función de activación (debe ser implementada por clases hijas)
   * @param input Valor de entrada
   */
  public abstract backward(input: number): number;

  /**
   * Aplicación vectorizada de la función de activación a una matriz
   * @param matrix Matriz de entrada
   */
  public forwardMatrix(matrix: number[][]): number[][] {
    const rows = matrix.length;
    const cols = matrix[0].length;
    const result: number[][] = Array(rows)
      .fill(0)
      .map(() => Array(cols).fill(0));

    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        result[i][j] = this.forward(matrix[i][j]);
      }
    }

    return result;
  }

  /**
   * Aplicación vectorizada de la derivada a una matriz
   * @param matrix Matriz de entrada
   */
  public backwardMatrix(matrix: number[][]): number[][] {
    const rows = matrix.length;
    const cols = matrix[0].length;
    const result: number[][] = Array(rows)
      .fill(0)
      .map(() => Array(cols).fill(0));

    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        result[i][j] = this.backward(matrix[i][j]);
      }
    }

    return result;
  }
}
