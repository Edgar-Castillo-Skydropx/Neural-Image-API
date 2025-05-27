/**
 * Interfaz que define un optimizador para el entrenamiento de redes neuronales
 */
export interface IOptimizer {
  // Propiedades
  readonly name: string;
  readonly learningRate: number;
  
  // Método para actualizar los pesos basado en gradientes
  updateWeights(weights: number[][], gradients: number[][]): number[][];
  
  // Métodos para configuración
  setLearningRate(learningRate: number): void;
  
  // Métodos para serialización
  toJSON(): Record<string, any>;
  fromJSON(config: Record<string, any>): void;
}
