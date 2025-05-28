import { Request, Response } from "express";
import { TrainingService } from "@/api/services/TrainingService";

/**
 * Controlador para gestionar las operaciones relacionadas con el entrenamiento
 */
export class TrainingController {
  private trainingService: TrainingService;

  /**
   * Constructor del controlador de entrenamiento
   */
  constructor() {
    this.trainingService = new TrainingService();
  }

  /**
   * Entrena la red neuronal con imágenes etiquetadas
   * @param req Solicitud HTTP
   * @param res Respuesta HTTP
   */
  public trainNetwork = async (req: Request, res: Response): Promise<void> => {
    try {
      if (!req.files || !Array.isArray(req.files) || req.files.length === 0) {
        res.status(400).json({
          success: false,
          error: {
            message: "No se han proporcionado imágenes para el entrenamiento",
            code: 400,
          },
        });
        return;
      }

      // Obtener parámetros de entrenamiento
      const { epochs, learningRate, batchSize, labels } = req.body;

      if (!labels) {
        res.status(400).json({
          success: false,
          error: {
            message:
              "Se requieren etiquetas para las imágenes de entrenamiento",
            code: 400,
          },
        });
        return;
      }

      // Preparar imágenes y etiquetas para entrenamiento
      const imagePaths = (req.files as Express.Multer.File[]).map(
        (file) => file.path
      );

      // Iniciar entrenamiento
      const trainingId = await this.trainingService.startTraining({
        imagePaths,
        labels: JSON.parse(labels),
        epochs: parseInt(epochs) || 10,
        learningRate: parseFloat(learningRate) || 0.01,
        batchSize: parseInt(batchSize) || 32,
      });

      res.status(200).json({
        success: true,
        data: {
          message: "Entrenamiento iniciado correctamente",
          trainingId,
        },
      });
    } catch (error) {
      console.error("Error al iniciar entrenamiento:", error);
      res.status(500).json({
        success: false,
        error: {
          message:
            error instanceof Error
              ? error.message
              : "Error al iniciar el entrenamiento",
          code: 500,
        },
      });
    }
  };

  /**
   * Obtiene el estado del entrenamiento
   * @param req Solicitud HTTP
   * @param res Respuesta HTTP
   */
  public getTrainingStatus = async (
    req: Request,
    res: Response
  ): Promise<void> => {
    try {
      const { trainingId } = req.query;

      if (!trainingId) {
        res.status(400).json({
          success: false,
          error: {
            message: "Se requiere el ID de entrenamiento",
            code: 400,
          },
        });
        return;
      }

      const status = await this.trainingService.getTrainingStatus(
        trainingId as string
      );

      res.status(200).json({
        success: true,
        data: status,
      });
    } catch (error) {
      console.error("Error al obtener estado del entrenamiento:", error);
      res.status(500).json({
        success: false,
        error: {
          message:
            error instanceof Error
              ? error.message
              : "Error al obtener estado del entrenamiento",
          code: 500,
        },
      });
    }
  };

  /**
   * Guarda el modelo entrenado
   * @param req Solicitud HTTP
   * @param res Respuesta HTTP
   */
  public saveModel = async (req: Request, res: Response): Promise<void> => {
    try {
      const { modelName, description } = req.body;

      if (!modelName) {
        res.status(400).json({
          success: false,
          error: {
            message: "Se requiere un nombre para el modelo",
            code: 400,
          },
        });
        return;
      }

      const result = await this.trainingService.saveModel(
        modelName,
        description
      );

      res.status(200).json({
        success: true,
        data: {
          message: "Modelo guardado correctamente",
          modelId: result.modelId,
        },
      });
    } catch (error) {
      console.error("Error al guardar modelo:", error);
      res.status(500).json({
        success: false,
        error: {
          message:
            error instanceof Error
              ? error.message
              : "Error al guardar el modelo",
          code: 500,
        },
      });
    }
  };

  /**
   * Carga un modelo previamente entrenado
   * @param req Solicitud HTTP
   * @param res Respuesta HTTP
   */
  public loadModel = async (req: Request, res: Response): Promise<void> => {
    try {
      const { modelId } = req.body;

      if (!modelId) {
        res.status(400).json({
          success: false,
          error: {
            message: "Se requiere el ID del modelo a cargar",
            code: 400,
          },
        });
        return;
      }

      const result = await this.trainingService.loadModel(modelId);

      res.status(200).json({
        success: true,
        data: {
          message: "Modelo cargado correctamente",
          modelDetails: result,
        },
      });
    } catch (error) {
      console.error("Error al cargar modelo:", error);
      res.status(500).json({
        success: false,
        error: {
          message:
            error instanceof Error
              ? error.message
              : "Error al cargar el modelo",
          code: 500,
        },
      });
    }
  };
}
