import { Schema, model, Document } from "mongoose";

/**
 * Interfaz para el documento de datos de entrenamiento en MongoDB
 */
export interface ITrainingDataDocument extends Document {
  _id: string;
  name: string;
  description?: string;
  status: "pending" | "in_progress" | "completed" | "failed";
  progress: number;
  images: {
    path: string;
    label: string;
    processed: boolean;
  }[];
  configuration: {
    epochs: number;
    batchSize: number;
    learningRate: number;
    optimizer: string;
  };
  results: {
    accuracy: number[];
    loss: number[];
    validationAccuracy: number[];
    validationLoss: number[];
  };
  metadata: {
    startTime?: Date;
    endTime?: Date;
    duration?: number;
    modelId?: string;
  };
}

/**
 * Esquema para los datos de entrenamiento en MongoDB
 */
const TrainingDataSchema = new Schema<ITrainingDataDocument>(
  {
    name: {
      type: String,
      required: true,
      trim: true,
    },
    description: {
      type: String,
      trim: true,
    },
    status: {
      type: String,
      required: true,
      enum: ["pending", "in_progress", "completed", "failed"],
      default: "pending",
    },
    progress: {
      type: Number,
      required: true,
      min: 0,
      max: 100,
      default: 0,
    },
    images: [
      {
        path: {
          type: String,
          required: true,
        },
        label: {
          type: String,
          required: true,
        },
        processed: {
          type: Boolean,
          default: false,
        },
      },
    ],
    configuration: {
      epochs: {
        type: Number,
        required: true,
        default: 10,
      },
      batchSize: {
        type: Number,
        required: true,
        default: 32,
      },
      learningRate: {
        type: Number,
        required: true,
        default: 0.01,
      },
      optimizer: {
        type: String,
        required: true,
        default: "sgd",
      },
    },
    results: {
      accuracy: [Number],
      loss: [Number],
      validationAccuracy: [Number],
      validationLoss: [Number],
    },
    metadata: {
      startTime: Date,
      endTime: Date,
      duration: Number,
      modelId: String,
    },
  },
  {
    timestamps: true,
  }
);

// Crear y exportar el modelo
export const TrainingData = model<ITrainingDataDocument>(
  "TrainingData",
  TrainingDataSchema
);
