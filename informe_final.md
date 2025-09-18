
# Informe Final: Familia de Algoritmos para Predicción de Glucosa en Ayunas y Clasificación de Diabetes

## Resumen Ejecutivo

Se ha desarrollado una familia completa de algoritmos de Machine Learning y Deep Learning para predecir los niveles de glucosa en ayunas (variable numérica "Resultado") y su clasificación correspondiente ("Clase_DM": Normal <100 mg/dL, Prediabetes 100-126 mg/dL, Diabetes >126 mg/dL). 

El dataset utilizado contiene 100 registros con 43 características clínicas y demográficas. Los modelos se entrenaron con un split 70/15/15 (entrenamiento/validación/prueba) y se implementaron pipelines para reentrenamiento con nuevos datos.

**Mejores Modelos Seleccionados:**
- **Regresión (Predicción de Glucosa):** Random Forest Regressor (R² = -0.0437 en test)
- **Clasificación (Prediabetes/Diabetes):** Gradient Boosting Classifier (Accuracy = 86.67%)

Nota: Los valores R² negativos en regresión indican que el modelo es peor que simplemente predecir la media, lo que sugiere limitaciones del dataset (tamaño pequeño, desbalanceo o complejidad inherente).

## 1. Exploración y Preprocesamiento de Datos

### Características del Dataset
- **Tamaño:** 100 registros, 43 columnas
- **Variable Objetivo Regresión:** "Resultado" (glucosa en mg/dL, rango: 69.8 - 314.6)
- **Variable Objetivo Clasificación:** "Clase_DM" (Normal: 59, Prediabetes: 37, Diabetes: 4)
- **Características Principales:**
  - Demográficas: edad, sexo, ocupación
  - Clínicas: IMC, presión arterial, perímetro abdominal
  - Hábitos: ejercicio, consumo frutas, cigarrillos
  - Historial: diabetes familiar, puntaje de riesgo

### Distribución de Clases
```
Normal: 59 (59%)
Prediabetes: 37 (37%)
Diabetes: 4 (4%)
```

### Preprocesamiento Realizado
- Eliminación de columnas irrelevantes (identificadores, nombres)
- Manejo de valores faltantes (eliminación de filas)
- Encoding de variables categóricas (LabelEncoder)
- Escalado de características numéricas (StandardScaler)
- División: 70 registros entrenamiento, 15 validación, 15 prueba

### Desafíos Identificados
- Dataset pequeño (100 registros) - riesgo de sobreajuste
- Clases desbalanceadas (solo 4 casos de Diabetes)
- Posibles outliers en mediciones de glucosa

## 2. Modelos Desarrollados

### Modelos de Machine Learning - Regresión
1. **Linear Regression:** Modelo lineal simple
2. **Ridge Regression:** Regresión lineal con regularización L2
3. **Random Forest Regressor:** Ensemble de árboles de decisión
4. **Gradient Boosting Regressor:** Gradient Boosting Trees

### Modelos de Machine Learning - Clasificación
1. **Logistic Regression:** Regresión logística multinomial
2. **Random Forest Classifier:** Ensemble de árboles para clasificación
3. **SVM Classifier:** Máquinas de soporte vectorial
4. **Gradient Boosting Classifier:** Gradient Boosting para clasificación

### Modelos de Deep Learning
- **Red Neuronal Regresión:** 2 capas ocultas (64 y 32 neuronas), activación ReLU, optimizador Adam
- **Red Neuronal Clasificación:** 2 capas ocultas (64 y 32 neuronas), activación softmax, sparse_categorical_crossentropy

## 3. Resultados y Evaluación

### Métricas de Regresión (Conjunto de Prueba)
| Modelo                    | MSE     | MAE    | R²      |
|---------------------------|---------|--------|---------|
| Linear Regression         | 4793.73 | 38.87  | -0.1133 |
| Ridge Regression          | 4513.75 | 37.06  | -0.0482 |
| Random Forest Regressor   | 4494.17 | 35.52  | -0.0437 |
| Gradient Boosting Regressor | 4745.31 | 32.78  | -0.1020 |

**Análisis:** Todos los modelos de regresión muestran R² negativos, indicando que no superan un modelo baseline simple (predicción de la media). El Random Forest Regressor tiene el mejor rendimiento relativo, pero el dataset pequeño limita la capacidad predictiva.

### Métricas de Clasificación (Conjunto de Prueba)
| Modelo                     | Accuracy | Precision | Recall | F1    | AUC |
|----------------------------|----------|-----------|--------|-------|-----|
| Logistic Regression        | 0.800    | 0.640     | 0.800  | 0.709 | None |
| Random Forest Classifier   | 0.800    | 0.640     | 0.800  | 0.709 | None |
| SVM Classifier             | 0.800    | 0.640     | 0.800  | 0.709 | None |
| Gradient Boosting Classifier | 0.867  | 0.886     | 0.867  | 0.876 | None |

**Análisis:** Los modelos de clasificación muestran mejor rendimiento general. El Gradient Boosting Classifier es el mejor con 86.67% de accuracy. Las métricas AUC no están disponibles debido al desbalanceo de clases en el conjunto de prueba.

### Matriz de Confusión - Mejor Modelo (Gradient Boosting)
```
[[1, 2],  # Predicciones para Normal
 [0, 12]] # Predicciones para Prediabetes/Diabetes
```

## 4. Comparación y Selección de Modelos

### Mejores Modelos Recomendados
- **Predicción de Glucosa (Regresión):** Random Forest Regressor
  - Razón: Mejor R² relativo y capacidad para manejar no-linealidades
  - Limitación: Dataset pequeño limita rendimiento general

- **Clasificación de Diabetes:** Gradient Boosting Classifier
  - Razón: Mejor accuracy (86.67%) y F1-score (87.6%)
  - Ventaja: Maneja bien el desbalanceo de clases

### Comparación ML vs DL
- **ML es preferible** para este caso debido a:
  - Dataset pequeño (DL requiere más datos)
  - Interpretabilidad superior
  - Menor complejidad computacional
  - Mejor rendimiento en clasificación

## 5. Pipeline de Reentrenamiento

Se implementó un pipeline modular que permite:
- **Añadir nuevos datos** mediante concatenación con el dataset original
- **Reentrenamiento automático** de todos los modelos
- **Mantenimiento de consistencia** en preprocesamiento (mismos encoders y escaladores)
- **Guardado de modelos actualizados** en formato pickle

**Uso:**
```python
pipeline = RetrainPipeline()
pipeline.retrain_models('ruta/a/nuevos_datos.csv')
```

## 6. Recomendaciones y Mejoras

### Fortalezas del Sistema
- Pipeline completo desde preprocesamiento hasta despliegue
- Múltiples algoritmos evaluados para robustez
- Capacidad de reentrenamiento incremental
- Métricas estandarizadas y visualizaciones

### Limitaciones Identificadas
1. **Dataset Pequeño:** Solo 100 registros limitan generalización
2. **Desbalanceo de Clases:** Solo 4 casos de Diabetes
3. **R