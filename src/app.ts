import express, { Application } from "express";
import cors from "cors";
import dotenv from "dotenv/lib/main";
import { imageRoutes } from "@/api/routes/imageRoutes";
import { trainingRoutes } from "@/api/routes/trainingRoutes";
import { errorHandler } from "@/api/middlewares/errorHandler";

// Cargar variables de entorno
dotenv.config();

/**
 * Clase principal de la aplicación
 * Configura y gestiona la aplicación Express
 */
class App {
  public app: Application;
  public port: number;

  /**
   * Constructor de la aplicación
   * @param port Puerto en el que se ejecutará el servidor
   */
  constructor(port: number) {
    this.app = express();
    this.port = port;

    this.initializeMiddlewares();
    this.initializeRoutes();
    this.initializeErrorHandling();
  }

  /**
   * Inicializa los middlewares globales
   */
  private initializeMiddlewares(): void {
    this.app.use(express.json());
    this.app.use(express.urlencoded({ extended: true }));
    this.app.use(cors());
  }

  /**
   * Inicializa las rutas de la API
   */
  private initializeRoutes(): void {
    this.app.use("/api/images", imageRoutes);
    this.app.use("/api/training", trainingRoutes);
  }

  /**
   * Inicializa el manejo de errores
   */
  private initializeErrorHandling(): void {
    this.app.use(errorHandler);
  }

  /**
   * Inicia el servidor
   */
  public listen(): void {
    this.app.listen(this.port, () => {
      console.log(`Servidor ejecutándose en http://localhost:${this.port}`);
    });
  }
}

export default App;
