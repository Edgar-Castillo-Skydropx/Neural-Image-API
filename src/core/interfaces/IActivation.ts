/**
 * Interfaz que define las funciones de activación para redes neuronales
 * Cada función de activación debe implementar tanto la función directa como su derivada
 */
export interface IActivation {
  // Nombre de la función de activación
  readonly name: string;

  // Función de activación directa
  // Transforma la entrada aplicando la función de activación
  forward(input: number): number;

  // Derivada de la función de activación
  // Necesaria para el algoritmo de retropropagación
  backward(input: number): number;

  // Aplicación vectorizada de la función de activación a una matriz
  // Aplica la función a cada elemento de la matriz
  forwardMatrix(matrix: number[][]): number[][];

  // Aplicación vectorizada de la derivada a una matriz
  // Aplica la derivada a cada elemento de la matriz
  backwardMatrix(matrix: number[][]): number[][];
}
