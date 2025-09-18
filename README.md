# Delfos A1 C7: Modelos ML/DL para Predicción de Glucosa y Clasificación de Diabetes

## Descripción
Proyecto para desarrollar una familia de algoritmos de Machine Learning y Deep Learning que predice los niveles de glucosa en ayunas ('Resultado') y clasifica el estado de diabetes ('Clase_DM': Normal <100 mg/dL, Prediabetes 100-126 mg/dL, Diabetes >126 mg/dL).

El dataset contiene 100 registros con 43 características clínicas y demográficas (edad, IMC, presión arterial, hábitos, historial familiar).

**Mejores Modelos:**
- **Regresión:** Random Forest Regressor (R² = -0.0437 en test)
- **Clasificación:** Gradient Boosting Classifier (Accuracy = 86.67%, F1 = 0.876)

Nota: R² negativos en regresión indican limitaciones del dataset pequeño/desbalanceado. Clasificación es más robusta para screening.

## Instalación
1. Clona el repositorio:
   ```
   git clone https://github.com/jhofeloto/delfosA1C7.git
   cd delfosA1C7
   ```

2. Instala dependencias (Python 3.9+):
   ```
   pip install scikit-learn pandas numpy matplotlib seaborn joblib tensorflow
   ```
   - Para TensorFlow en macOS M1/M2: `conda install tensorflow` si hay conflictos.

## Estructura del Proyecto
- `data/`: Dataset original (`output-glucosa_labeled.csv`).
- `models/`: Modelos entrenados (.pkl para ML).
- `results/`: Métricas JSON y visualizaciones PNG (MSE comparison, accuracy bars, confusion matrix).
- `preprocessed_data.npz`: Datos divididos (70/15/15).
- Scripts principales:
  - `explore_dataset.py`: Análisis inicial y visualizaciones.
  - `preprocess_data.py`: Limpieza, encoding, escalado.
  - `train_ml_models.py`: Entrenamiento ML (regresión/clasificación).
  - `train_dl_models.py`: Entrenamiento DL (redes neuronales).
  - `compare_models.py`: Comparación, selección de mejores, gráficos.
  - `retrain_pipeline.py`: Pipeline para reentrenamiento con nuevos datos.

## Uso
### 1. Exploración Inicial
```
python explore_dataset.py
```
Genera histogramas de glucosa y matriz de correlación (ver PNGs en root/results/).

### 2. Preprocesamiento
```
python preprocess_data.py
```
Crea `preprocessed_data.npz` con datos escalados y divididos.

### 3. Entrenamiento ML
```
python train_ml_models.py
```
Entrena 4 modelos regresión + 4 clasificación. Modelos en `models/`, métricas en `results/ml_results.json`.

### 4. Entrenamiento DL
```
python train_dl_models.py
```
Entrena redes neuronales. Modelos en `models/` (.h5), métricas en `results/dl_results.json` (nota: puede fallar en algunos entornos por compatibilidad TensorFlow).

### 5. Comparación y Visualizaciones
```
python compare_models.py
```
Genera tablas de métricas, gráficos (MSE/Accuracy/confusión) y selecciona mejores modelos. Resultados en `results/model_comparison.json` y PNGs.

### 6. Reentrenamiento con Nuevos Datos
```
python retrain_pipeline.py
```
Para datos nuevos:
```python
from retrain_pipeline import RetrainPipeline
pipeline = RetrainPipeline()
pipeline.retrain_models('path/to/new_data.csv')  # Opcional; usa original si None
```
Añade datos, reentrena ML, guarda modelos actualizados. Mantiene encoders/escaladores consistentes.

## Resultados Destacados
### Métricas Regresión (Test)
| Modelo | MSE | MAE | R² |
|--------|-----|-----|----|
| Random Forest | 4494.17 | 35.52 | -0.0437 (mejor) |

### Métricas Clasificación (Test)
| Modelo | Accuracy | F1 |
|--------|----------|----|
| Gradient Boosting | 0.867 | 0.876 (mejor) |

Ver `informe_final.md` para análisis completo, matrices de confusión y recomendaciones (e.g., más datos para mejorar R² >0).

## Requisitos y Notas
- **Dependencias:** scikit-learn, pandas, numpy, matplotlib, seaborn, joblib, tensorflow.
- **Limitaciones:** Dataset pequeño (100 registros); DL puede fallar en macOS/Anaconda (usa conda env). Desbalanceo (4 Diabetes) causa warnings.
- **Mejoras Pendientes:** Deployment (API FastAPI), validación cruzada, SHAP para features.

## Contribución
- Fork el repo, crea branch, commit cambios, PR a main.
- Para datos nuevos: Añade CSV compatible y usa pipeline.

Proyecto desarrollado con 12/12 tareas completadas. ¡Gracias por usar Delfos A1 C7!