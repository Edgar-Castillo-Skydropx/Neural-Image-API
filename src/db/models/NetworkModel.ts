import { Schema, model, Document } from 'mongoose';

/**
 * Interfaz para el documento de modelo de red neuronal en MongoDB
 */
export interface INetworkModelDocument extends Document {
  name: string;
  description?: string;
  architecture: string;
  layers: {
    id: string;
    type: string;
    weights: Record<string, any>;
  }[];
  performance: {
    accuracy?: number;
    loss?: number;
    validationAccuracy?: number;
    validationLoss?: number;
  };
  metadata: {
    createdAt: Date;
    updatedAt: Date;
    version: string;
    trainingTime?: number;
    epochs?: number;
  };
}

/**
 * Esquema para el modelo de red neuronal en MongoDB
 */
const NetworkModelSchema = new Schema<INetworkModelDocument>({
  name: {
    type: String,
    required: true,
    unique: true,
    trim: true
  },
  description: {
    type: String,
    trim: true
  },
  architecture: {
    type: String,
    required: true,
    enum: ['sequential', 'convolutional']
  },
  layers: [{
    id: {
      type: String,
      required: true
    },
    type: {
      type: String,
      required: true
    },
    weights: {
      type: Schema.Types.Mixed,
      required: true
    }
  }],
  performance: {
    accuracy: Number,
    loss: Number,
    validationAccuracy: Number,
    validationLoss: Number
  },
  metadata: {
    createdAt: {
      type: Date,
      default: Date.now
    },
    updatedAt: {
      type: Date,
      default: Date.now
    },
    version: {
      type: String,
      default: '1.0.0'
    },
    trainingTime: Number,
    epochs: Number
  }
}, {
  timestamps: true
});

// Middleware para actualizar el campo updatedAt
NetworkModelSchema.pre('save', function(next) {
  this.metadata.updatedAt = new Date();
  next();
});

// Crear y exportar el modelo
export const NetworkModel = model<INetworkModelDocument>('NetworkModel', NetworkModelSchema);
