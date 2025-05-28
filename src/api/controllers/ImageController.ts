import { Request, Response } from "express";
import path from "path";
import fs from "fs";
import { ImageService } from "@/api/services/ImageService";

/**
 * Controlador para gestionar las operaciones relacionadas con imágenes
 */
export class ImageController {
  private imageService: ImageService;

  /**
   * Constructor del controlador de imágenes
   */
  constructor() {
    this.imageService = new ImageService();
  }

  /**
   * Clasifica una imagen subida
   * @param req Solicitud HTTP
   * @param res Respuesta HTTP
   */
  public classifyImage = async (req: Request, res: Response): Promise<void> => {
    try {
      if (!req.file) {
        res.status(400).json({
          success: false,
          error: {
            message: "No se ha proporcionado ninguna imagen",
            code: 400,
          },
        });
        return;
      }

      const imagePath = req.file.path;

      // Procesar y clasificar la imagen
      const result = await this.imageService.classifyImage(imagePath);

      res.status(200).json({
        success: true,
        data: {
          classification: result.classification,
          confidence: result.confidence,
          processingTime: result.processingTime,
        },
      });
    } catch (error) {
      console.error("Error al clasificar imagen:", error);
      res.status(500).json({
        success: false,
        error: {
          message:
            error instanceof Error
              ? error.message
              : "Error al procesar la imagen",
          code: 500,
        },
      });
    }
  };

  /**
   * Obtiene el estado del servicio de clasificación
   * @param req Solicitud HTTP
   * @param res Respuesta HTTP
   */
  public getStatus = async (req: Request, res: Response): Promise<void> => {
    try {
      const status = await this.imageService.getServiceStatus();

      res.status(200).json({
        success: true,
        data: {
          status: status.isActive ? "active" : "inactive",
          modelLoaded: status.modelLoaded,
          lastUpdated: status.lastUpdated,
          version: status.version,
        },
      });
    } catch (error) {
      console.error("Error al obtener estado del servicio:", error);
      res.status(500).json({
        success: false,
        error: {
          message:
            error instanceof Error
              ? error.message
              : "Error al obtener estado del servicio",
          code: 500,
        },
      });
    }
  };
}
