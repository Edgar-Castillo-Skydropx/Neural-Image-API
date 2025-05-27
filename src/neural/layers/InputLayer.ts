import { BaseLayer } from "@/neural/layers/BaseLayer";

/**
 * Implementación de una capa de entrada para la red neuronal
 * Simplemente pasa los datos de entrada sin modificarlos
 */
export class InputLayer extends BaseLayer {
  /**
   * Constructor de la capa de entrada
   * @param id Identificador único de la capa
   * @param inputShape Forma de entrada [filas, columnas, canales]
   */
  constructor(id: string, inputShape: number[]) {
    super(id, "input", inputShape, inputShape, undefined);
  }

  /**
   * Inicializa la capa (no requiere inicialización)
   */
  public initialize(): void {
    // No se requiere inicialización para la capa de entrada
  }

  /**
   * Propagación hacia adelante (simplemente pasa los datos)
   * @param input Datos de entrada
   * @returns Los mismos datos de entrada
   */
  public forward(input: number[][]): number[][] {
    return input;
  }

  /**
   * Retropropagación (simplemente pasa el gradiente)
   * @param outputGradient Gradiente de salida
   * @param learningRate Tasa de aprendizaje (no utilizada)
   * @returns El mismo gradiente de salida
   */
  public backward(
    outputGradient: number[][],
    learningRate: number
  ): number[][] {
    return outputGradient;
  }

  /**
   * Obtiene los pesos de la capa (no tiene pesos)
   */
  public getWeights(): Record<string, number[][]> {
    return {};
  }

  /**
   * Establece los pesos de la capa (no tiene pesos)
   * @param weights Pesos a establecer (ignorados)
   */
  public setWeights(weights: Record<string, number[][]>): void {
    // No hay pesos que establecer
  }

  /**
   * Carga la configuración de la capa desde formato JSON
   * @param config Configuración en formato JSON
   */
  public fromJSON(config: Record<string, any>): void {
    // No hay configuración que cargar
  }
}
