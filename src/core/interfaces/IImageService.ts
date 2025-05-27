/**
 * Interfaz para el servicio de procesamiento de imágenes
 */
export interface IImageService {
  /**
   * Clasifica una imagen
   * @param imagePath Ruta de la imagen a clasificar
   * @returns Resultado de la clasificación
   */
  classifyImage(imagePath: string): Promise<{
    classification: string;
    confidence: number;
    processingTime: number;
  }>;

  /**
   * Obtiene el estado del servicio de clasificación
   * @returns Estado del servicio
   */
  getServiceStatus(): Promise<{
    isActive: boolean;
    modelLoaded: boolean;
    lastUpdated: Date | null;
    version: string;
  }>;
}
