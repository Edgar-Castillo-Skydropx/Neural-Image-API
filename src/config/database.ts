import mongoose from "mongoose";
import dotenv from "dotenv";

// Cargar variables de entorno
dotenv.config();

/**
 * Clase para gestionar la conexión a la base de datos MongoDB
 */
export class Database {
  private static instance: Database;
  private isConnected: boolean = false;

  /**
   * Constructor privado para implementar patrón Singleton
   */
  private constructor() {}

  /**
   * Obtiene la instancia única de la base de datos
   */
  public static getInstance(): Database {
    if (!Database.instance) {
      Database.instance = new Database();
    }
    return Database.instance;
  }

  /**
   * Conecta a la base de datos MongoDB
   */
  public async connect(): Promise<void> {
    if (this.isConnected) {
      console.log("Ya existe una conexión a la base de datos");
      return;
    }

    try {
      const mongoUri =
        process.env.MONGODB_URI || "mongodb://localhost:27017/neural-image-api";

      await mongoose.connect(mongoUri, {
        // Opciones de conexión
      });

      this.isConnected = true;
      console.log("Conexión a MongoDB establecida correctamente");
    } catch (error) {
      console.error("Error al conectar a MongoDB:", error);
      throw error;
    }
  }

  /**
   * Desconecta de la base de datos MongoDB
   */
  public async disconnect(): Promise<void> {
    if (!this.isConnected) {
      console.log("No hay conexión activa a la base de datos");
      return;
    }

    try {
      await mongoose.disconnect();
      this.isConnected = false;
      console.log("Desconexión de MongoDB realizada correctamente");
    } catch (error) {
      console.error("Error al desconectar de MongoDB:", error);
      throw error;
    }
  }

  /**
   * Verifica si hay una conexión activa a la base de datos
   */
  public isConnectedToDatabase(): boolean {
    return this.isConnected;
  }
}
