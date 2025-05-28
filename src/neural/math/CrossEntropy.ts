/**
 * Implementación de la función de pérdida Cross-Entropy para clasificación multiclase
 *
 * Cross-Entropy es la función de pérdida más adecuada para problemas de clasificación
 * multiclase cuando se usa con activación softmax en la capa de salida.
 *
 * La fórmula matemática es:
 * L = -∑(y_true * log(y_pred))
 * donde y_true son las etiquetas verdaderas (one-hot) y y_pred son las probabilidades predichas
 */
export class CrossEntropy {
  /**
   * Calcula la pérdida Cross-Entropy entre las predicciones y los objetivos
   * @param predictions Predicciones del modelo (después de softmax) [batch_size, num_classes]
   * @param targets Objetivos reales (codificación one-hot) [batch_size, num_classes]
   * @returns Valor escalar de pérdida
   */
  public static loss(predictions: number[][], targets: number[][]): number {
    let totalLoss = 0;
    const epsilon = 1e-15; // Pequeño valor para evitar log(0)

    for (let i = 0; i < predictions.length; i++) {
      let sampleLoss = 0;

      for (let j = 0; j < predictions[i].length; j++) {
        // Clip para evitar log(0)
        const clippedPred = Math.max(
          Math.min(predictions[i][j], 1 - epsilon),
          epsilon
        );
        sampleLoss -= targets[i][j] * Math.log(clippedPred);
      }

      totalLoss += sampleLoss;
    }

    // Normalizar por el tamaño del batch
    return totalLoss / predictions.length;
  }

  /**
   * Calcula el gradiente de la pérdida Cross-Entropy respecto a las predicciones
   *
   * Cuando se combina con softmax, el gradiente se simplifica a (predictions - targets)
   * Esta simplificación es una propiedad matemática de la combinación softmax + cross-entropy
   *
   * @param predictions Predicciones del modelo (después de softmax) [batch_size, num_classes]
   * @param targets Objetivos reales (codificación one-hot) [batch_size, num_classes]
   * @returns Gradiente [batch_size, num_classes]
   */
  public static gradient(
    predictions: number[][],
    targets: number[][]
  ): number[][] {
    const gradient: number[][] = [];

    for (let i = 0; i < predictions.length; i++) {
      gradient[i] = [];

      for (let j = 0; j < predictions[i].length; j++) {
        // El gradiente de softmax + cross-entropy es simplemente (pred - target)
        gradient[i][j] = predictions[i][j] - targets[i][j];
      }
    }

    // Normalizar por el tamaño del batch
    for (let i = 0; i < gradient.length; i++) {
      for (let j = 0; j < gradient[i].length; j++) {
        gradient[i][j] /= predictions.length;
      }
    }

    return gradient;
  }
}
