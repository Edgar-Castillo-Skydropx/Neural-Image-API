import { TrainingData, ITrainingDataDocument } from "@/db/models/TrainingData";

/**
 * Repositorio para gestionar operaciones con datos de entrenamiento en la base de datos
 */
export class TrainingRepository {
  /**
   * Crea un nuevo registro de entrenamiento
   * @param trainingData Datos del entrenamiento a crear
   * @returns Documento del entrenamiento creado
   */
  public async createTraining(trainingData: {
    name: string;
    description?: string;
    images: {
      path: string;
      label: string;
    }[];
    configuration: {
      epochs: number;
      batchSize: number;
      learningRate: number;
      optimizer: string;
      imageSize: number;
    };
  }): Promise<ITrainingDataDocument> {
    try {
      const newTraining = new TrainingData({
        name: trainingData.name,
        description: trainingData.description,
        status: "pending",
        progress: 0,
        images: trainingData.images.map((img) => ({
          path: img.path,
          label: img.label,
          processed: false,
        })),
        configuration: trainingData.configuration,
        results: {
          accuracy: [],
          loss: [],
          validationAccuracy: [],
          validationLoss: [],
        },
        metadata: {},
      });

      await newTraining.save();
      return newTraining;
    } catch (error) {
      console.error("Error al crear registro de entrenamiento:", error);
      throw error;
    }
  }

  /**
   * Obtiene un registro de entrenamiento por su ID
   * @param trainingId ID del entrenamiento a obtener
   * @returns Documento del entrenamiento encontrado
   */
  public async getTrainingById(
    trainingId: string
  ): Promise<ITrainingDataDocument | null> {
    try {
      return await TrainingData.findById(trainingId);
    } catch (error) {
      console.error("Error al obtener entrenamiento por ID:", error);
      throw error;
    }
  }

  /**
   * Actualiza el estado de un entrenamiento
   * @param trainingId ID del entrenamiento a actualizar
   * @param status Nuevo estado
   * @param progress Progreso actual (0-100)
   * @returns Documento del entrenamiento actualizado
   */
  public async updateTrainingStatus(
    trainingId: string,
    status: "pending" | "in_progress" | "completed" | "failed",
    progress: number
  ): Promise<ITrainingDataDocument | null> {
    try {
      const updates: Record<string, any> = { status, progress };

      // Si el entrenamiento está comenzando
      if (status === "in_progress" && progress === 0) {
        updates["metadata.startTime"] = new Date();
      }

      // Si el entrenamiento ha finalizado
      if (status === "completed" || status === "failed") {
        const endTime = new Date();
        updates["metadata.endTime"] = endTime;

        // Calcular duración si hay tiempo de inicio
        const training = await TrainingData.findById(trainingId);
        if (training && training.metadata.startTime) {
          const startTime = training.metadata.startTime;
          const duration = (endTime.getTime() - startTime.getTime()) / 1000; // en segundos
          updates["metadata.duration"] = duration;
        }
      }

      return await TrainingData.findByIdAndUpdate(trainingId, updates, {
        new: true,
      });
    } catch (error) {
      console.error("Error al actualizar estado del entrenamiento:", error);
      throw error;
    }
  }

  /**
   * Actualiza los resultados de un entrenamiento
   * @param trainingId ID del entrenamiento a actualizar
   * @param results Resultados del entrenamiento
   * @returns Documento del entrenamiento actualizado
   */
  public async updateTrainingResults(
    trainingId: string,
    results: {
      accuracy: number[];
      loss: number[];
      validationAccuracy?: number[];
      validationLoss?: number[];
    }
  ): Promise<ITrainingDataDocument | null> {
    try {
      return await TrainingData.findByIdAndUpdate(
        trainingId,
        { results },
        { new: true }
      );
    } catch (error) {
      console.error("Error al actualizar resultados del entrenamiento:", error);
      throw error;
    }
  }

  /**
   * Asocia un modelo a un entrenamiento
   * @param trainingId ID del entrenamiento
   * @param modelId ID del modelo generado
   * @returns Documento del entrenamiento actualizado
   */
  public async associateModelToTraining(
    trainingId: string,
    modelId: string
  ): Promise<ITrainingDataDocument | null> {
    try {
      return await TrainingData.findByIdAndUpdate(
        trainingId,
        { "metadata.modelId": modelId },
        { new: true }
      );
    } catch (error) {
      console.error("Error al asociar modelo al entrenamiento:", error);
      throw error;
    }
  }

  /**
   * Obtiene todos los entrenamientos
   * @returns Lista de documentos de entrenamientos
   */
  public async getAllTrainings(): Promise<ITrainingDataDocument[]> {
    try {
      return await TrainingData.find().sort({ createdAt: -1 });
    } catch (error) {
      console.error("Error al obtener todos los entrenamientos:", error);
      throw error;
    }
  }

  /**
   * Elimina un entrenamiento por su ID
   * @param trainingId ID del entrenamiento a eliminar
   * @returns true si se eliminó correctamente
   */
  public async deleteTraining(trainingId: string): Promise<boolean> {
    try {
      const result = await TrainingData.deleteOne({ _id: trainingId });
      return result.deletedCount > 0;
    } catch (error) {
      console.error("Error al eliminar entrenamiento:", error);
      throw error;
    }
  }
}
