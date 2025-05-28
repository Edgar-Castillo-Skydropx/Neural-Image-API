import { IModel } from "../../core/interfaces/IModel";
import { ILayer } from "../../core/interfaces/ILayer";
import { IOptimizer } from "../../core/interfaces/IOptimizer";
/**
 * Clase base abstracta para todos los modelos de red neuronal
 * Implementa la interfaz IModel y proporciona funcionalidad común
 */
export abstract class BaseModel implements IModel {
  public readonly id: string;
  public readonly name: string;
  public readonly layers: string[];
  protected layerInstances: ILayer[];
  protected optimizer: IOptimizer | null;
  protected isInitialized: boolean;

  /**
   * Constructor del modelo base
   * @param id Identificador único del modelo
   * @param name Nombre del modelo
   */
  constructor(id: string, name: string) {
    this.id = id;
    this.name = name;
    this.layers = [];
    this.layerInstances = [];
    this.optimizer = null;
    this.isInitialized = false;
  }

  /**
   * Inicializa el modelo y sus capas
   */
  public initialize(): void {
    if (this.isInitialized) {
      return;
    }

    // Inicializar cada capa
    for (const layer of this.layerInstances) {
      layer.initialize();
    }

    this.isInitialized = true;
  }

  /**
   * Realiza una predicción con el modelo
   * @param input Datos de entrada
   */
  public predict(input: number[][]): number[][] {
    if (!this.isInitialized) {
      this.initialize();
    }

    let output = input;

    // Propagación hacia adelante a través de todas las capas
    for (const layer of this.layerInstances) {
      output = layer.forward(output);
    }

    return output;
  }

  /**
   * Entrena el modelo con datos de entrada y salida esperada
   * @param inputs Conjunto de datos de entrada
   * @param targets Salidas esperadas correspondientes
   * @param epochs Número de épocas de entrenamiento
   * @param batchSize Tamaño del lote para entrenamiento
   */
  public abstract train(
    inputs: number[][][],
    targets: number[][][],
    epochs: number,
    batchSize: number
  ): Promise<Record<string, number[]>>;

  /**
   * Evalúa el rendimiento del modelo
   * @param inputs Conjunto de datos de entrada para evaluación
   * @param targets Salidas esperadas correspondientes
   */
  public abstract evaluate(
    inputs: number[][][],
    targets: number[][][]
  ): Record<string, number>;

  /**
   * Guarda el modelo en formato serializable
   */
  public save(): Record<string, any> {
    const layersData = this.layerInstances.map((layer) => ({
      id: layer.id,
      type: layer.type,
      weights: layer.getWeights(),
    }));

    return {
      id: this.id,
      name: this.name,
      layers: this.layers,
      layersData,
      optimizer: this.optimizer ? this.optimizer.toJSON() : null,
    };
  }

  /**
   * Carga el modelo desde formato serializado
   * @param modelData Datos del modelo serializado
   */
  public load(modelData: Record<string, any>): void {
    if (!modelData.layersData || !Array.isArray(modelData.layersData)) {
      throw new Error("Datos de modelo inválidos: falta información de capas");
    }

    // Cargar datos en cada capa
    for (let i = 0; i < this.layerInstances.length; i++) {
      const layerData = modelData.layersData.find(
        (ld: any) => ld.id === this.layerInstances[i].id
      );

      if (layerData && layerData.weights) {
        this.layerInstances[i].setWeights(layerData.weights);
      }
    }

    this.isInitialized = true;
  }
}
