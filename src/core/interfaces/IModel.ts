/**
 * Interfaz que define un modelo de red neuronal
 */
export interface IModel {
  // Propiedades
  readonly id: string;
  readonly name: string;
  readonly layers: string[];
  
  // Métodos de inicialización
  initialize(): void;
  
  // Métodos de predicción
  predict(input: number[][]): number[][];
  
  // Métodos de entrenamiento
  train(
    inputs: number[][][], 
    targets: number[][][], 
    epochs: number, 
    batchSize: number
  ): Promise<Record<string, number[]>>;
  
  // Métodos para evaluación
  evaluate(inputs: number[][][], targets: number[][][]): Record<string, number>;
  
  // Métodos para guardar y cargar
  save(): Record<string, any>;
  load(modelData: Record<string, any>): void;
}
