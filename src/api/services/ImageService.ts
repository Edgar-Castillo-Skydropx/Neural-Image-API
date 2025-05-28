import { IImageService } from "@/core/interfaces/IImageService";
import { IModel } from "@/core/interfaces/IModel";
import { SequentialModel } from "@/neural/models/SequentialModel";
import { ConvolutionalModel } from "@/neural/models/ConvolutionalModel";
import { ModelRepository } from "@/db/repositories/ModelRepository";
import fs from "fs";
import { promisify } from "util";
import { createCanvas, loadImage } from "canvas";

// Convertir fs.readFile a promesa
const readFile = promisify(fs.readFile);

// Clases disponibles para clasificación
// En un entorno real, esto podría cargarse desde la base de datos o un archivo de configuración
const CLASSES = [
  "avión",
  "automóvil",
  "pájaro",
  "gato",
  "ciervo",
  "perro",
  "rana",
  "caballo",
  "barco",
  "camión",
];

/**
 * Servicio para procesar y clasificar imágenes
 */
export class ImageService implements IImageService {
  private model: IModel | null = null;
  private modelRepository: ModelRepository;
  private isModelLoaded: boolean = false;
  private lastModelUpdate: Date | null = null;
  private classes: string[] = CLASSES;
  private imageSize: number = 32; // Tamaño estándar para imágenes (ej. 32x32 para CIFAR-10)

  /**
   * Constructor del servicio de imágenes
   */
  constructor() {
    this.modelRepository = new ModelRepository();
  }

  /**
   * Carga el modelo más reciente desde la base de datos
   */
  private async loadLatestModel(): Promise<void> {
    try {
      // Obtener todos los modelos ordenados por fecha de actualización
      const models = await this.modelRepository.getAllModels();

      if (models.length === 0) {
        console.log("No hay modelos disponibles para cargar");
        return;
      }

      // Tomar el modelo más reciente
      const latestModel = models[0];

      // Crear instancia del modelo según la arquitectura
      if (latestModel.architecture === "sequential") {
        this.model = new SequentialModel(
          latestModel._id.toString(),
          latestModel.name
        );
      } else if (latestModel.architecture === "convolutional") {
        this.model = new ConvolutionalModel(
          latestModel._id.toString(),
          latestModel.name
        );
      } else {
        throw new Error(
          `Arquitectura de modelo no soportada: ${latestModel.architecture}`
        );
      }

      // Cargar pesos y configuración
      this.model.load({
        id: latestModel._id.toString(),
        name: latestModel.name,
        layersData: latestModel.layers,
      });

      // Si el modelo tiene metadatos con clases, cargarlas
      if (latestModel.metadata && latestModel.metadata.classes) {
        this.classes = latestModel.metadata.classes;
      }

      // Si el modelo tiene metadatos con tamaño de imagen, cargarlo
      if (latestModel.metadata && latestModel.metadata.imageSize) {
        this.imageSize = latestModel.metadata.imageSize;
      }

      this.isModelLoaded = true;
      this.lastModelUpdate = latestModel.metadata.updatedAt;

      console.log(`Modelo cargado: ${latestModel.name}`);
      console.log(`Clases disponibles: ${this.classes.join(", ")}`);
    } catch (error) {
      console.error("Error al cargar el modelo:", error);
      throw error;
    }
  }

  /**
   * Preprocesa una imagen para la clasificación
   * @param imagePath Ruta de la imagen a preprocesar
   * @returns Matriz de datos de la imagen preprocesada
   */
  private async preprocessImage(imagePath: string): Promise<number[][]> {
    try {
      // Verificar que el archivo existe
      if (!fs.existsSync(imagePath)) {
        throw new Error(`Archivo de imagen no encontrado: ${imagePath}`);
      }

      // Cargar la imagen usando la biblioteca canvas
      const image = await loadImage(imagePath);

      // Crear un canvas del tamaño requerido por el modelo
      const canvas = createCanvas(this.imageSize, this.imageSize);
      const ctx = canvas.getContext("2d");

      // Dibujar la imagen redimensionada en el canvas
      ctx.drawImage(image, 0, 0, this.imageSize, this.imageSize);

      // Obtener los datos de píxeles
      const imageData = ctx.getImageData(0, 0, this.imageSize, this.imageSize);
      const pixels = imageData.data;

      // Normalizar y aplanar los datos de la imagen
      // Para un modelo convolucional, necesitamos mantener la estructura espacial
      // Para un modelo secuencial, aplanamos completamente
      const isConvolutional = this.model instanceof ConvolutionalModel;

      if (isConvolutional) {
        // Para modelo convolucional: formato [batch_size, height, width, channels]
        // Normalizar valores a rango [0,1]
        const normalizedData: number[][][] = [];
        const imageArray: number[][] = [];

        for (let y = 0; y < this.imageSize; y++) {
          const row: number[] = [];
          for (let x = 0; x < this.imageSize; x++) {
            const pos = (y * this.imageSize + x) * 4; // RGBA = 4 canales

            // Promedio de RGB para escala de grises o usar los 3 canales
            // Aquí usamos escala de grises para simplificar
            const r = pixels[pos] / 255;
            const g = pixels[pos + 1] / 255;
            const b = pixels[pos + 2] / 255;
            const grayScale = (r + g + b) / 3;

            row.push(grayScale);
          }
          imageArray.push(row);
        }

        // Aplanar para formato de entrada del modelo
        const flattenedInput: number[] = [];
        for (let y = 0; y < this.imageSize; y++) {
          for (let x = 0; x < this.imageSize; x++) {
            flattenedInput.push(imageArray[y][x]);
          }
        }

        return [flattenedInput]; // Formato [batch_size=1, flattened_image]
      } else {
        // Para modelo secuencial: aplanar completamente
        const flattenedInput: number[] = [];

        // Normalizar valores a rango [0,1]
        for (let i = 0; i < pixels.length; i += 4) {
          const r = pixels[i] / 255;
          const g = pixels[i + 1] / 255;
          const b = pixels[i + 2] / 255;

          // Usar escala de grises o los 3 canales separados
          // Aquí usamos escala de grises para simplificar
          const grayScale = (r + g + b) / 3;
          flattenedInput.push(grayScale);
        }

        return [flattenedInput]; // Formato [batch_size=1, flattened_image]
      }
    } catch (error) {
      console.error("Error al preprocesar imagen:", error);
      throw error;
    }
  }

  /**
   * Clasifica una imagen
   * @param imagePath Ruta de la imagen a clasificar
   * @returns Resultado de la clasificación
   */
  public async classifyImage(imagePath: string): Promise<{
    classification: string;
    confidence: number;
    processingTime: number;
    topPredictions?: Array<{ class: string; probability: number }>;
  }> {
    try {
      // Verificar si el modelo está cargado
      if (!this.isModelLoaded || !this.model) {
        await this.loadLatestModel();

        // Si aún no hay modelo disponible
        if (!this.isModelLoaded || !this.model) {
          throw new Error("No hay modelo disponible para clasificación");
        }
      }

      const startTime = Date.now();

      // Preprocesar imagen
      const preprocessedImage = await this.preprocessImage(imagePath);

      // Realizar predicción
      const prediction = this.model.predict(preprocessedImage);

      // Interpretar la salida del modelo para determinar la clase y confianza
      // La salida típica de un modelo de clasificación es un vector de probabilidades
      // donde cada índice corresponde a una clase
      const outputVector = prediction[0]; // Primera (y única) muestra del batch

      // Encontrar el índice con la probabilidad más alta
      let maxIndex = 0;
      let maxProbability = outputVector[0];

      for (let i = 1; i < outputVector.length; i++) {
        if (outputVector[i] > maxProbability) {
          maxProbability = outputVector[i];
          maxIndex = i;
        }
      }

      // Obtener la clase correspondiente al índice
      let classification = "desconocido";
      if (maxIndex < this.classes.length) {
        classification = this.classes[maxIndex];
      }

      // Calcular el tiempo de procesamiento
      const processingTime = (Date.now() - startTime) / 1000; // en segundos

      // Obtener las N predicciones principales (top-N)
      const topN = 3; // Número de predicciones principales a devolver
      const indexedProbabilities = outputVector.map((prob, idx) => ({
        index: idx,
        probability: prob,
      }));

      // Ordenar por probabilidad descendente
      indexedProbabilities.sort((a, b) => b.probability - a.probability);

      // Tomar las top-N predicciones
      const topPredictions = indexedProbabilities
        .slice(0, topN)
        .map((item) => ({
          class:
            item.index < this.classes.length
              ? this.classes[item.index]
              : `clase_${item.index}`,
          probability: item.probability,
        }));

      // Resultado final
      const result = {
        classification,
        confidence: maxProbability,
        processingTime,
        topPredictions,
      };

      return result;
    } catch (error) {
      console.error("Error al clasificar imagen:", error);
      throw error;
    }
  }

  /**
   * Obtiene el estado del servicio de clasificación
   * @returns Estado del servicio
   */
  public async getServiceStatus(): Promise<{
    isActive: boolean;
    modelLoaded: boolean;
    lastUpdated: Date | null;
    version: string;
    supportedClasses?: string[];
  }> {
    return {
      isActive: true,
      modelLoaded: this.isModelLoaded,
      lastUpdated: this.lastModelUpdate,
      version: "1.0.0",
      supportedClasses: this.classes,
    };
  }
}
