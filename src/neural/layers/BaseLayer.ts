import { ILayer } from "@/core/interfaces/ILayer";
import { ActivationType } from "@/core/types/ActivationType";
import { IActivation } from "@/core/interfaces/IActivation";
import { ActivationFactory } from "@/neural/activations/ActivationFactory";

/**
 * Clase base abstracta para todas las capas neuronales
 * Implementa la interfaz ILayer y proporciona funcionalidad común
 */
export abstract class BaseLayer implements ILayer {
  public readonly id: string;
  public readonly type: string;
  public readonly inputShape: number[];
  public readonly outputShape: number[];
  protected activation: IActivation | null;

  /**
   * Constructor de la capa base
   * @param id Identificador único de la capa
   * @param type Tipo de capa
   * @param inputShape Forma de entrada [filas, columnas, canales]
   * @param outputShape Forma de salida [filas, columnas, canales]
   * @param activationType Tipo de función de activación (opcional)
   */
  constructor(
    id: string,
    type: string,
    inputShape: number[],
    outputShape: number[],
    activationType?: ActivationType
  ) {
    this.id = id;
    this.type = type;
    this.inputShape = [...inputShape];
    this.outputShape = [...outputShape];
    this.activation = activationType
      ? ActivationFactory.create(activationType)
      : null;
  }

  /**
   * Inicializa la capa (debe ser implementado por clases hijas)
   */
  public abstract initialize(): void;

  /**
   * Propagación hacia adelante (debe ser implementado por clases hijas)
   * @param input Datos de entrada
   */
  public abstract forward(input: number[][]): number[][];

  /**
   * Retropropagación (debe ser implementado por clases hijas)
   * @param outputGradient Gradiente de salida
   * @param learningRate Tasa de aprendizaje
   */
  public abstract backward(
    outputGradient: number[][],
    learningRate: number
  ): number[][];

  /**
   * Obtiene los pesos de la capa
   */
  public abstract getWeights(): Record<string, number[][]>;

  /**
   * Establece los pesos de la capa
   * @param weights Pesos a establecer
   */
  public abstract setWeights(weights: Record<string, number[][]>): void;

  /**
   * Convierte la capa a formato JSON para serialización
   */
  public toJSON(): Record<string, any> {
    return {
      id: this.id,
      type: this.type,
      inputShape: this.inputShape,
      outputShape: this.outputShape,
      activation: this.activation ? this.activation.name : null,
      weights: this.getWeights(),
    };
  }

  /**
   * Carga la configuración de la capa desde formato JSON
   * @param config Configuración en formato JSON
   */
  public abstract fromJSON(config: Record<string, any>): void;
}
