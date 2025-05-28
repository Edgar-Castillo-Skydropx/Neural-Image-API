import { NetworkModel, INetworkModelDocument } from "@/db/models/NetworkModel";

/**
 * Repositorio para gestionar operaciones con modelos de red neuronal en la base de datos
 */
export class ModelRepository {
  /**
   * Guarda un nuevo modelo en la base de datos
   * @param modelData Datos del modelo a guardar
   * @returns Documento del modelo guardado
   */
  public async saveModel(modelData: {
    name: string;
    description?: string;
    architecture: string;
    layers: {
      id: string;
      type: string;
      weights: Record<string, any>;
    }[];
    performance?: {
      accuracy?: number;
      loss?: number;
      validationAccuracy?: number;
      validationLoss?: number;
    };
    metadata?: {
      version?: string;
      trainingTime?: number;
      epochs?: number;
      classes?: string[];
      imageSize?: number;
    };
  }): Promise<INetworkModelDocument> {
    try {
      // Verificar si ya existe un modelo con el mismo nombre
      const existingModel = await NetworkModel.findOne({
        name: modelData.name,
      });

      if (existingModel) {
        // Actualizar modelo existente
        Object.assign(existingModel, {
          description: modelData.description,
          architecture: modelData.architecture,
          layers: modelData.layers,
          performance: modelData.performance || existingModel.performance,
          "metadata.updatedAt": new Date(),
          "metadata.version":
            modelData.metadata?.version || existingModel.metadata.version,
          "metadata.trainingTime":
            modelData.metadata?.trainingTime ||
            existingModel.metadata.trainingTime,
          "metadata.epochs":
            modelData.metadata?.epochs || existingModel.metadata.epochs,
          "metadata.classes":
            modelData.metadata?.classes || existingModel.metadata.classes,
          "metadata.imageSize":
            modelData.metadata?.imageSize || existingModel.metadata.imageSize,
        });

        await existingModel.save();
        return existingModel;
      } else {
        // Crear nuevo modelo
        const newModel = new NetworkModel({
          name: modelData.name,
          description: modelData.description,
          architecture: modelData.architecture,
          layers: modelData.layers,
          performance: modelData.performance || {
            accuracy: 0,
            loss: 0,
          },
          metadata: {
            createdAt: new Date(),
            updatedAt: new Date(),
            version: modelData.metadata?.version || "1.0.0",
            trainingTime: modelData.metadata?.trainingTime,
            epochs: modelData.metadata?.epochs,
            classes: modelData.metadata?.classes,
            imageSize: modelData.metadata?.imageSize,
          },
        });

        await newModel.save();
        return newModel;
      }
    } catch (error) {
      console.error("Error al guardar modelo en la base de datos:", error);
      throw error;
    }
  }

  /**
   * Obtiene un modelo por su ID
   * @param modelId ID del modelo a obtener
   * @returns Documento del modelo encontrado
   */
  public async getModelById(
    modelId: string
  ): Promise<INetworkModelDocument | null> {
    try {
      return await NetworkModel.findById(modelId);
    } catch (error) {
      console.error("Error al obtener modelo por ID:", error);
      throw error;
    }
  }

  /**
   * Obtiene un modelo por su nombre
   * @param name Nombre del modelo a obtener
   * @returns Documento del modelo encontrado
   */
  public async getModelByName(
    name: string
  ): Promise<INetworkModelDocument | null> {
    try {
      return await NetworkModel.findOne({ name });
    } catch (error) {
      console.error("Error al obtener modelo por nombre:", error);
      throw error;
    }
  }

  /**
   * Obtiene todos los modelos disponibles
   * @returns Lista de documentos de modelos
   */
  public async getAllModels(): Promise<INetworkModelDocument[]> {
    try {
      return await NetworkModel.find().sort({ "metadata.updatedAt": -1 });
    } catch (error) {
      console.error("Error al obtener todos los modelos:", error);
      throw error;
    }
  }

  /**
   * Elimina un modelo por su ID
   * @param modelId ID del modelo a eliminar
   * @returns true si se eliminó correctamente
   */
  public async deleteModel(modelId: string): Promise<boolean> {
    try {
      const result = await NetworkModel.deleteOne({ _id: modelId });
      return result.deletedCount > 0;
    } catch (error) {
      console.error("Error al eliminar modelo:", error);
      throw error;
    }
  }

  /**
   * Actualiza el rendimiento de un modelo
   * @param modelId ID del modelo a actualizar
   * @param performance Datos de rendimiento actualizados
   * @returns Documento del modelo actualizado
   */
  public async updateModelPerformance(
    modelId: string,
    performance: {
      accuracy?: number;
      loss?: number;
      validationAccuracy?: number;
      validationLoss?: number;
    }
  ): Promise<INetworkModelDocument | null> {
    try {
      return await NetworkModel.findByIdAndUpdate(
        modelId,
        {
          performance,
          "metadata.updatedAt": new Date(),
        },
        { new: true }
      );
    } catch (error) {
      console.error("Error al actualizar rendimiento del modelo:", error);
      throw error;
    }
  }

  /**
   * Actualiza los metadatos de un modelo
   * @param modelId ID del modelo a actualizar
   * @param metadata Metadatos actualizados
   * @returns Documento del modelo actualizado
   */
  public async updateModelMetadata(
    modelId: string,
    metadata: {
      version?: string;
      trainingTime?: number;
      epochs?: number;
      classes?: string[];
      imageSize?: number;
    }
  ): Promise<INetworkModelDocument | null> {
    try {
      const updateData: Record<string, any> = {
        "metadata.updatedAt": new Date(),
      };

      // Añadir solo los campos proporcionados
      if (metadata.version !== undefined) {
        updateData["metadata.version"] = metadata.version;
      }
      if (metadata.trainingTime !== undefined) {
        updateData["metadata.trainingTime"] = metadata.trainingTime;
      }
      if (metadata.epochs !== undefined) {
        updateData["metadata.epochs"] = metadata.epochs;
      }
      if (metadata.classes !== undefined) {
        updateData["metadata.classes"] = metadata.classes;
      }
      if (metadata.imageSize !== undefined) {
        updateData["metadata.imageSize"] = metadata.imageSize;
      }

      return await NetworkModel.findByIdAndUpdate(modelId, updateData, {
        new: true,
      });
    } catch (error) {
      console.error("Error al actualizar metadatos del modelo:", error);
      throw error;
    }
  }
}
