/**
 * Interfaz que define la estructura básica de una capa neuronal
 */
export interface ILayer {
  // Propiedades
  readonly id: string;
  readonly type: string;
  readonly inputShape: number[];
  readonly outputShape: number[];

  // Métodos de inicialización
  initialize(): void;

  // Propagación hacia adelante
  forward(input: number[][]): number[][];

  // Retropropagación
  backward(outputGradient: number[][], learningRate: number): number[][];

  // Métodos para guardar y cargar pesos
  getWeights(): Record<string, number[][]>;
  setWeights(weights: Record<string, number[][]>): void;

  // Métodos para serialización
  toJSON(): Record<string, any>;
  fromJSON(config: Record<string, any>): void;
}
