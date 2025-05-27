import { IOptimizer } from "@/core/interfaces/IOptimizer";

/**
 * Clase base abstracta para todos los optimizadores
 * Implementa la interfaz IOptimizer y proporciona funcionalidad común
 */
export abstract class BaseOptimizer implements IOptimizer {
  public readonly name: string;
  public readonly learningRate: number;

  /**
   * Constructor del optimizador base
   * @param name Nombre del optimizador
   * @param learningRate Tasa de aprendizaje
   */
  constructor(name: string, learningRate: number = 0.01) {
    this.name = name;
    this.learningRate = learningRate;
  }

  /**
   * Actualiza los pesos basado en gradientes (debe ser implementado por clases hijas)
   * @param weights Pesos actuales
   * @param gradients Gradientes calculados
   */
  public abstract updateWeights(
    weights: number[][],
    gradients: number[][]
  ): number[][];

  /**
   * Establece una nueva tasa de aprendizaje
   * @param learningRate Nueva tasa de aprendizaje
   */
  public setLearningRate(learningRate: number): void {
    (this as any).learningRate = learningRate;
  }

  /**
   * Convierte el optimizador a formato JSON para serialización
   */
  public toJSON(): Record<string, any> {
    return {
      name: this.name,
      learningRate: this.learningRate,
    };
  }

  /**
   * Carga la configuración del optimizador desde formato JSON
   * @param config Configuración en formato JSON
   */
  public fromJSON(config: Record<string, any>): void {
    if (config.learningRate) {
      this.setLearningRate(config.learningRate);
    }
  }
}
