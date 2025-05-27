import { IMatrix } from "@/core/interfaces/IMatrix";

/**
 * Implementación de la interfaz IMatrix para operaciones matriciales
 * Base matemática fundamental para la red neuronal
 */
export class Matrix implements IMatrix {
  public readonly rows: number;
  public readonly cols: number;
  public readonly data: number[][];

  /**
   * Constructor de la matriz
   * @param rows Número de filas
   * @param cols Número de columnas
   * @param initializer Función inicializadora opcional
   */
  constructor(
    rows: number,
    cols: number,
    initializer?: (row: number, col: number) => number
  ) {
    this.rows = rows;
    this.cols = cols;
    this.data = Array(rows)
      .fill(0)
      .map((_, i) =>
        Array(cols)
          .fill(0)
          .map((_, j) => (initializer ? initializer(i, j) : 0))
      );
  }

  /**
   * Crea una matriz a partir de un array bidimensional
   * @param array Array bidimensional de números
   */
  public static fromArray(array: number[][]): Matrix {
    if (array.length === 0 || array[0].length === 0) {
      throw new Error("Array cannot be empty");
    }

    const rows = array.length;
    const cols = array[0].length;

    // Verificar que todas las filas tengan la misma longitud
    for (let i = 0; i < rows; i++) {
      if (array[i].length !== cols) {
        throw new Error("All rows must have the same length");
      }
    }

    const matrix = new Matrix(rows, cols);
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        matrix.data[i][j] = array[i][j];
      }
    }

    return matrix;
  }

  /**
   * Crea una matriz con valores aleatorios
   * @param rows Número de filas
   * @param cols Número de columnas
   * @param min Valor mínimo (por defecto -1)
   * @param max Valor máximo (por defecto 1)
   */
  public static random(
    rows: number,
    cols: number,
    min: number = -1,
    max: number = 1
  ): Matrix {
    return new Matrix(rows, cols, () => min + Math.random() * (max - min));
  }

  /**
   * Crea una matriz de ceros
   * @param rows Número de filas
   * @param cols Número de columnas
   */
  public static zeros(rows: number, cols: number): Matrix {
    return new Matrix(rows, cols);
  }

  /**
   * Crea una matriz de unos
   * @param rows Número de filas
   * @param cols Número de columnas
   */
  public static ones(rows: number, cols: number): Matrix {
    return new Matrix(rows, cols, () => 1);
  }

  /**
   * Crea una matriz identidad
   * @param size Tamaño de la matriz (filas = columnas)
   */
  public static identity(size: number): Matrix {
    return new Matrix(size, size, (i, j) => (i === j ? 1 : 0));
  }

  /**
   * Obtiene el valor en una posición específica
   * @param row Índice de fila
   * @param col Índice de columna
   */
  public get(row: number, col: number): number {
    this.validateIndices(row, col);
    return this.data[row][col];
  }

  /**
   * Establece un valor en una posición específica
   * @param row Índice de fila
   * @param col Índice de columna
   * @param value Valor a establecer
   */
  public set(row: number, col: number, value: number): void {
    this.validateIndices(row, col);
    this.data[row][col] = value;
  }

  /**
   * Crea una copia de la matriz
   */
  public clone(): Matrix {
    const clone = new Matrix(this.rows, this.cols);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        clone.data[i][j] = this.data[i][j];
      }
    }
    return clone;
  }

  /**
   * Suma esta matriz con otra
   * @param matrix Matriz a sumar
   */
  public add(matrix: IMatrix): Matrix {
    this.validateDimensions(matrix);

    const result = new Matrix(this.rows, this.cols);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        result.data[i][j] = this.data[i][j] + matrix.data[i][j];
      }
    }

    return result;
  }

  /**
   * Resta otra matriz de esta
   * @param matrix Matriz a restar
   */
  public subtract(matrix: IMatrix): Matrix {
    this.validateDimensions(matrix);

    const result = new Matrix(this.rows, this.cols);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        result.data[i][j] = this.data[i][j] - matrix.data[i][j];
      }
    }

    return result;
  }

  /**
   * Multiplica esta matriz por otra (producto matricial)
   * @param matrix Matriz a multiplicar
   */
  public multiply(matrix: IMatrix): Matrix {
    if (this.cols !== matrix.rows) {
      throw new Error(
        `Cannot multiply matrices of dimensions ${this.rows}x${this.cols} and ${matrix.rows}x${matrix.cols}`
      );
    }

    const result = new Matrix(this.rows, matrix.cols);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < matrix.cols; j++) {
        let sum = 0;
        for (let k = 0; k < this.cols; k++) {
          sum += this.data[i][k] * matrix.data[k][j];
        }
        result.data[i][j] = sum;
      }
    }

    return result;
  }

  /**
   * Realiza el producto de Hadamard (multiplicación elemento a elemento)
   * @param matrix Matriz para el producto de Hadamard
   */
  public hadamardProduct(matrix: IMatrix): Matrix {
    this.validateDimensions(matrix);

    const result = new Matrix(this.rows, this.cols);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        result.data[i][j] = this.data[i][j] * matrix.data[i][j];
      }
    }

    return result;
  }

  /**
   * Transpone la matriz
   */
  public transpose(): Matrix {
    const result = new Matrix(this.cols, this.rows);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        result.data[j][i] = this.data[i][j];
      }
    }

    return result;
  }

  /**
   * Multiplica la matriz por un escalar
   * @param scalar Valor escalar
   */
  public multiplyScalar(scalar: number): Matrix {
    const result = new Matrix(this.rows, this.cols);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        result.data[i][j] = this.data[i][j] * scalar;
      }
    }

    return result;
  }

  /**
   * Suma un escalar a cada elemento de la matriz
   * @param scalar Valor escalar
   */
  public addScalar(scalar: number): Matrix {
    const result = new Matrix(this.rows, this.cols);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        result.data[i][j] = this.data[i][j] + scalar;
      }
    }

    return result;
  }

  /**
   * Aplica una función a cada elemento de la matriz
   * @param callback Función a aplicar
   */
  public map(
    callback: (value: number, row: number, col: number) => number
  ): Matrix {
    const result = new Matrix(this.rows, this.cols);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        result.data[i][j] = callback(this.data[i][j], i, j);
      }
    }

    return result;
  }

  /**
   * Ejecuta una función para cada elemento de la matriz
   * @param callback Función a ejecutar
   */
  public forEach(
    callback: (value: number, row: number, col: number) => void
  ): void {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        callback(this.data[i][j], i, j);
      }
    }
  }

  /**
   * Convierte la matriz a un array bidimensional
   */
  public toArray(): number[][] {
    return this.data.map((row) => [...row]);
  }

  /**
   * Convierte la matriz a una cadena de texto
   */
  public toString(): string {
    return this.data.map((row) => row.join("\t")).join("\n");
  }

  /**
   * Valida que los índices estén dentro de los límites
   * @param row Índice de fila
   * @param col Índice de columna
   */
  private validateIndices(row: number, col: number): void {
    if (row < 0 || row >= this.rows || col < 0 || col >= this.cols) {
      throw new Error(
        `Index out of bounds: (${row}, ${col}) for matrix of size ${this.rows}x${this.cols}`
      );
    }
  }

  /**
   * Valida que las dimensiones de otra matriz coincidan con esta
   * @param matrix Matriz a validar
   */
  private validateDimensions(matrix: IMatrix): void {
    if (this.rows !== matrix.rows || this.cols !== matrix.cols) {
      throw new Error(
        `Matrix dimensions do not match: ${this.rows}x${this.cols} and ${matrix.rows}x${matrix.cols}`
      );
    }
  }
}
