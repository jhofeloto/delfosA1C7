# Delfos A1 C7: Modelos ML/DL para Predicción de Glucosa y Clasificación de Diabetes

## 🌐 Aplicación Web Desplegada
**🚀 URL de Acceso:** https://5000-iz225t8a71i9xfs3i2n2k-6532622b.e2b.dev

## Descripción
Sistema completo de predicción inteligente que combina algoritmos avanzados de Machine Learning y Deep Learning para predecir niveles de glucosa en ayunas ('Resultado') y clasificar el estado de diabetes ('Clase_DM': Normal <100 mg/dL, Prediabetes 100-126 mg/dL, Diabetes >126 mg/dL).

El sistema incluye una aplicación web completa con interfaz interactiva, visualizaciones dinámicas y API para predicciones en tiempo real.

**Dataset:** 100 registros con 43 características clínicas y demográficas (edad, IMC, presión arterial, hábitos, historial familiar).

**Mejores Modelos:**
- **Regresión:** Random Forest Regressor (R² = -0.0437 en test)
- **Clasificación:** Gradient Boosting Classifier (Accuracy = 86.67%, F1 = 0.876)

**Nota:** R² negativos en regresión indican limitaciones del dataset pequeño/desbalanceado. Clasificación es más robusta para screening.

## 🚀 Acceso Rápido
### Aplicación Web (Recomendado)
Accede directamente a la aplicación web desplegada: **https://5000-iz225t8a71i9xfs3i2n2k-6532622b.e2b.dev**

Características de la aplicación web:
- 🔮 **Predicciones en tiempo real** de glucosa y diabetes
- 📊 **Visualizaciones interactivas** con Plotly
- 📈 **Dashboard de métricas** de modelos
- 🏥 **Interfaz médica profesional** con interpretaciones
- 📱 **Responsive design** para cualquier dispositivo

### Instalación Local
1. Clona el repositorio:
   ```bash
   git clone https://github.com/jhofeloto/delfosA1C7.git
   cd delfosA1C7
   ```

2. Instala dependencias (Python 3.9+):
   ```bash
   pip install -r requirements.txt
   ```
   
   O instala manualmente:
   ```bash
   pip install flask gunicorn plotly pandas numpy scikit-learn matplotlib seaborn joblib supervisor
   ```
   - Para TensorFlow (opcional): `pip install tensorflow` o `conda install tensorflow` en macOS M1/M2.

3. **Ejecutar aplicación web:**
   ```bash
   # Opción 1: Desarrollo
   python app.py
   
   # Opción 2: Producción con Supervisor
   supervisord -c supervisord.conf
   supervisorctl -c supervisord.conf status
   ```

## Estructura del Proyecto

### 🌐 Aplicación Web
- **`app.py`**: Aplicación Flask principal con integración ML
- **`templates/`**: Plantillas HTML completas
  - `index.html`: Página de inicio con overview del proyecto
  - `predict.html`: Formulario interactivo de predicciones
  - `results.html`: Dashboard de métricas de modelos
  - `visualizations.html`: Gráficos interactivos y análisis
  - `about.html`: Documentación detallada del proyecto
- **`requirements.txt`**: Dependencias Python
- **`supervisord.conf`**: Configuración de deployment

### 📊 Sistema ML/DL
- **`data/`**: Dataset original (`output-glucosa_labeled.csv`)
- **`models/`**: Modelos entrenados (.pkl para ML)
- **`results/`**: Métricas JSON y visualizaciones PNG
- **`preprocessed_data.npz`**: Datos procesados y divididos (70/15/15)

### 🔬 Scripts de Análisis
- **`explore_dataset.py`**: Análisis inicial y visualizaciones
- **`preprocess_data.py`**: Limpieza, encoding, escalado
- **`train_ml_models.py`**: Entrenamiento ML (regresión/clasificación)
- **`train_dl_models.py`**: Entrenamiento DL (redes neuronales)
- **`compare_models.py`**: Comparación y selección de mejores modelos
- **`retrain_pipeline.py`**: Pipeline para reentrenamiento con nuevos datos

## Uso

### 🌐 Aplicación Web (Recomendado)
1. **Accede a la aplicación:** https://5000-iz225t8a71i9xfs3i2n2k-6532622b.e2b.dev
2. **Hacer Predicciones:**
   - Ve a la sección "Predicciones"
   - Ingresa los datos clínicos del paciente
   - Obtén predicciones instantáneas de glucosa y clasificación de diabetes
3. **Explorar Resultados:**
   - Revisa las métricas de los modelos en "Resultados"
   - Visualiza gráficos interactivos en "Visualizaciones"
   - Lee la documentación completa en "Acerca de"

### 📊 API Endpoints
- **POST /api/predict**: Predicciones programáticas
- **GET /api/results_data**: Métricas de modelos
- **GET /api/generate_chart/{tipo}**: Gráficos dinámicos

### 🔬 Scripts de Línea de Comandos
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

## 🌟 Características de la Aplicación Web

### 🔮 Predicciones Inteligentes
- **Predicción de Glucosa:** Random Forest Regressor con interpretación médica
- **Clasificación de Diabetes:** Gradient Boosting (86.7% accuracy) para Normal/Prediabetes/Diabetes
- **Interfaz Intuitiva:** Formulario con validación y guías médicas
- **Resultados Instantáneos:** Predicciones en tiempo real con interpretación automática

### 📊 Visualizaciones Avanzadas
- **Gráficos Interactivos:** Plotly.js para comparaciones de modelos
- **Dashboard de Métricas:** MSE, Accuracy, F1-Score, matrices de confusión
- **Análisis Visual:** Distribuciones, comparaciones de rendimiento
- **Imágenes Estáticas:** Visualizaciones del análisis original

### 🏥 Interfaz Médica Profesional
- **Disclaimers Médicos:** Advertencias apropiadas sobre uso clínico
- **Interpretación de Resultados:** Explicaciones claras para profesionales
- **Diseño Responsivo:** Acceso desde cualquier dispositivo
- **Navegación Intuitiva:** Estructura clara y profesional

### 🚀 Arquitectura Técnica
- **Flask Framework:** Aplicación web robusta y escalable
- **Gunicorn + Supervisor:** Deployment de producción con 4 workers
- **Bootstrap 5:** UI moderna y responsiva
- **API REST:** Endpoints para integración programática
- **Manejo de Errores:** Páginas personalizadas 404/500

## Requisitos y Notas
- **Dependencias:** scikit-learn, pandas, numpy, matplotlib, seaborn, joblib, tensorflow.
- **Limitaciones:** Dataset pequeño (100 registros); DL puede fallar en macOS/Anaconda (usa conda env). Desbalanceo (4 Diabetes) causa warnings.
- **✅ Deployment Completado:** Aplicación web Flask con API REST, interfaz profesional y visualizaciones interactivas.
- **Mejoras Futuras:** Validación cruzada, SHAP para interpretabilidad de features, integración con sistemas hospitalarios.

## Contribución
- Fork el repo, crea branch, commit cambios, PR a main.
- Para datos nuevos: Añade CSV compatible y usa pipeline.

Proyecto desarrollado con 12/12 tareas completadas. ¡Gracias por usar Delfos A1 C7!