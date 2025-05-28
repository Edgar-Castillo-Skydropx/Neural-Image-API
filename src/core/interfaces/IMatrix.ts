/**
 * Interfaz que define las operaciones básicas para matrices
 * Implementa operaciones matemáticas fundamentales para redes neuronales
 */
export interface IMatrix {
  // Propiedades
  readonly rows: number;
  readonly cols: number;
  readonly data: number[][];

  // Métodos básicos
  get(row: number, col: number): number;
  set(row: number, col: number, value: number): void;
  clone(): IMatrix;

  // Operaciones matriciales
  add(matrix: IMatrix): IMatrix;
  subtract(matrix: IMatrix): IMatrix;
  multiply(matrix: IMatrix): IMatrix;
  hadamardProduct(matrix: IMatrix): IMatrix; // Multiplicación elemento a elemento
  transpose(): IMatrix;

  // Operaciones con escalares
  multiplyScalar(scalar: number): IMatrix;
  addScalar(scalar: number): IMatrix;

  // Operaciones avanzadas
  map(callback: (value: number, row: number, col: number) => number): IMatrix;
  forEach(callback: (value: number, row: number, col: number) => void): void;

  // Utilidades
  toArray(): number[][];
  toString(): string;
}
