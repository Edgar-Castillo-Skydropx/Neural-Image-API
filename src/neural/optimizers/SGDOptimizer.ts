import { BaseOptimizer } from "@/neural/optimizers/BaseOptimizer";

/**
 * Implementación del optimizador SGD (Stochastic Gradient Descent)
 * Actualiza los pesos utilizando el descenso de gradiente estocástico
 */
export class SGDOptimizer extends BaseOptimizer {
  /**
   * Constructor del optimizador SGD
   * @param learningRate Tasa de aprendizaje
   */
  constructor(learningRate: number = 0.01) {
    super("sgd", learningRate);
  }

  /**
   * Actualiza los pesos basado en gradientes
   * @param weights Pesos actuales
   * @param gradients Gradientes calculados
   * @returns Pesos actualizados
   */
  public updateWeights(weights: number[][], gradients: number[][]): number[][] {
    const updatedWeights: number[][] = [];

    for (let i = 0; i < weights.length; i++) {
      updatedWeights[i] = [];
      for (let j = 0; j < weights[i].length; j++) {
        // w = w - lr * gradient
        updatedWeights[i][j] =
          weights[i][j] - this.learningRate * gradients[i][j];
      }
    }

    return updatedWeights;
  }
}
