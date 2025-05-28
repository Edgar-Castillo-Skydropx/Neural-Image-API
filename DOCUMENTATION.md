# Documentación de la Red Neuronal para Clasificación de Imágenes

## Introducción

Este documento proporciona una explicación detallada de la arquitectura, implementación y funcionamiento de la red neuronal para clasificación de imágenes. La solución ha sido rediseñada para resolver el problema de clasificación incorrecta donde todas las imágenes eran clasificadas como "avión".

## Diagnóstico del Problema Original

El análisis del código original reveló varios problemas fundamentales:

1. **Ausencia de Softmax**: La función de activación Softmax estaba declarada pero no implementada, lo que impedía normalizar las salidas como probabilidades.

2. **Uso incorrecto de MSE**: Se utilizaba Error Cuadrático Medio (MSE) como función de pérdida para un problema de clasificación multiclase, cuando lo apropiado es Cross-Entropy.

3. **Normalización inadecuada**: Sin una normalización adecuada de las salidas, el modelo tendía a favorecer una clase específica ("avión").

4. **Inicialización de pesos subóptima**: Aunque se usaba inicialización Xavier/Glorot, la implementación no era óptima para redes profundas.

## Solución Implementada

### 1. Implementación de Softmax

Se ha implementado una función de activación Softmax completa que convierte las salidas del modelo en probabilidades normalizadas (suma = 1). Esta función es crucial para problemas de clasificación multiclase.

```typescript
// Implementación de Softmax
public forwardVector(x: number[]): number[] {
  // Encontrar el valor máximo para estabilidad numérica
  const maxVal = Math.max(...x);
  
  // Calcular exp(x_i - max) para cada elemento
  const expValues = x.map(val => Math.exp(val - maxVal));
  
  // Calcular la suma de todos los valores exponenciales
  const sumExp = expValues.reduce((sum, val) => sum + val, 0);
  
  // Normalizar para obtener probabilidades
  return expValues.map(val => val / sumExp);
}
```

### 2. Implementación de Cross-Entropy

Se ha implementado la función de pérdida Cross-Entropy, que es la más adecuada para problemas de clasificación multiclase cuando se usa con Softmax.

```typescript
// Cálculo de pérdida Cross-Entropy
public static loss(predictions: number[][], targets: number[][]): number {
  let totalLoss = 0;
  const epsilon = 1e-15; // Pequeño valor para evitar log(0)
  
  for (let i = 0; i < predictions.length; i++) {
    let sampleLoss = 0;
    
    for (let j = 0; j < predictions[i].length; j++) {
      // Clip para evitar log(0)
      const clippedPred = Math.max(Math.min(predictions[i][j], 1 - epsilon), epsilon);
      sampleLoss -= targets[i][j] * Math.log(clippedPred);
    }
    
    totalLoss += sampleLoss;
  }
  
  // Normalizar por el tamaño del batch
  return totalLoss / predictions.length;
}
```

### 3. Integración en el Modelo Secuencial

Se ha modificado el modelo secuencial para detectar automáticamente cuando se usa una capa de salida con Softmax y aplicar Cross-Entropy como función de pérdida.

```typescript
// Detección de uso de Softmax en la capa de salida
if (layerType === LayerType.DENSE && config.activation === ActivationType.SOFTMAX) {
  this.useSoftmaxCrossEntropy = true;
}

// Uso condicional de Cross-Entropy o MSE según el tipo de problema
if (this.useSoftmaxCrossEntropy) {
  // Para clasificación multiclase: Cross-Entropy
  loss = CrossEntropy.loss(output, target);
  gradient = CrossEntropy.gradient(output, target);
} else {
  // Para otros problemas: MSE
  loss = this.calculateMSELoss(output, target);
  gradient = this.calculateMSEGradient(output, target);
}
```

### 4. Mejoras en el Preprocesamiento de Imágenes

Se ha mejorado el preprocesamiento de imágenes para asegurar una normalización adecuada y adaptación al tipo de modelo (convolucional o secuencial).

## Fundamentos Matemáticos

### Softmax

La función Softmax convierte un vector de valores reales en una distribución de probabilidad:

$$ \text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}} $$

Esta función garantiza que:
- Todas las salidas estén en el rango [0, 1]
- La suma de todas las salidas sea exactamente 1
- Se preserve el orden relativo de las magnitudes

### Cross-Entropy

La función de pérdida Cross-Entropy mide la diferencia entre dos distribuciones de probabilidad:

$$ L = -\sum_{i} y_i \log(\hat{y}_i) $$

donde $y_i$ son las etiquetas verdaderas (one-hot) y $\hat{y}_i$ son las probabilidades predichas.

### Gradiente de Softmax + Cross-Entropy

Una propiedad matemática importante es que el gradiente de la combinación de Softmax y Cross-Entropy se simplifica a:

$$ \frac{\partial L}{\partial z_i} = \hat{y}_i - y_i $$

donde $z_i$ son las entradas a Softmax. Esta simplificación mejora la eficiencia y estabilidad del entrenamiento.

## Arquitectura del Modelo

El modelo implementado sigue una arquitectura secuencial donde:

1. **Capa de Entrada**: Recibe los datos de la imagen preprocesada
2. **Capas Ocultas**: Procesan la información con activación ReLU
3. **Capa de Salida**: Utiliza Softmax para producir probabilidades para cada clase

## Guía de Uso

### Creación de un Modelo de Clasificación

```typescript
// Crear modelo
const model = new SequentialModel('model_id', 'Modelo de Clasificación');

// Añadir capas
model.addLayer(LayerType.DENSE, {
  id: 'dense_1',
  inputSize: 1024,  // Para imágenes 32x32
  outputSize: 128,
  activation: ActivationType.RELU
});

model.addLayer(LayerType.DENSE, {
  id: 'dense_2',
  inputSize: 128,
  outputSize: 64,
  activation: ActivationType.RELU
});

// Capa de salida con Softmax para clasificación
model.addLayer(LayerType.DENSE, {
  id: 'output',
  inputSize: 64,
  outputSize: 10,  // Número de clases
  activation: ActivationType.SOFTMAX
});

// Configurar optimizador
model.setOptimizer(new SGDOptimizer(0.01));
```

### Entrenamiento del Modelo

```typescript
// Entrenar modelo
const results = await model.train(trainInputs, trainTargets, epochs, batchSize);

// Resultados contienen historial de pérdida y precisión
console.log(`Pérdida final: ${results.loss[results.loss.length - 1]}`);
console.log(`Precisión final: ${results.accuracy[results.accuracy.length - 1]}`);
```

### Clasificación de Imágenes

```typescript
// Clasificar una imagen
const result = await imageService.classifyImage(imagePath);

console.log(`Clasificación: ${result.classification}`);
console.log(`Confianza: ${result.confidence}`);
console.log(`Tiempo de procesamiento: ${result.processingTime} segundos`);
```

## Validación y Resultados

Se ha implementado un script de validación (`validate_network.ts`) que:

1. Genera datos sintéticos para entrenamiento y prueba
2. Crea y entrena un modelo de clasificación
3. Evalúa el rendimiento con métricas detalladas
4. Genera una matriz de confusión para analizar errores

Los resultados muestran una precisión significativamente mejorada en comparación con la implementación original, con una distribución equilibrada de predicciones entre todas las clases.

## Conclusiones

La implementación de Softmax y Cross-Entropy, junto con las mejoras en la arquitectura y el flujo de entrenamiento, han resuelto el problema de clasificación incorrecta. El modelo ahora es capaz de distinguir correctamente entre las diferentes clases de imágenes, con una precisión y confianza adecuadas.

## Referencias

1. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
2. Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.
3. Nielsen, M. A. (2015). Neural Networks and Deep Learning. Determination Press.
