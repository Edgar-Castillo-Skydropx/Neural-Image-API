/**
 * Interfaz para el servicio de entrenamiento
 */
export interface ITrainingService {
  /**
   * Inicia un nuevo entrenamiento
   * @param options Opciones de entrenamiento
   * @returns ID del entrenamiento iniciado
   */
  startTraining(options: {
    imagePaths: string[];
    labels: string[];
    epochs?: number;
    learningRate?: number;
    batchSize?: number;
  }): Promise<string>;

  /**
   * Obtiene el estado de un entrenamiento
   * @param trainingId ID del entrenamiento
   * @returns Estado del entrenamiento
   */
  getTrainingStatus(trainingId: string): Promise<{
    status: string;
    progress: number;
    results?: {
      accuracy: number[];
      loss: number[];
    };
  }>;

  /**
   * Guarda un modelo entrenado
   * @param modelName Nombre del modelo
   * @param description Descripción opcional
   * @returns Información del modelo guardado
   */
  saveModel(
    modelName: string,
    description?: string
  ): Promise<{ modelId: string }>;

  /**
   * Carga un modelo previamente entrenado
   * @param modelId ID del modelo a cargar
   * @returns Detalles del modelo cargado
   */
  loadModel(modelId: string): Promise<{
    name: string;
    architecture: string;
    performance: {
      accuracy?: number;
      loss?: number;
    };
  }>;
}
