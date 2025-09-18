from flask import Flask, render_template, request, jsonify, send_file
import joblib
import numpy as np
import pandas as pd
import json
import os
import matplotlib
matplotlib.use('Agg')  # Para evitar problemas con la interfaz gráfica
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import plotly.graph_objs as go
import plotly.utils

app = Flask(__name__)

# Cargar modelos pre-entrenados
MODELS_DIR = 'models'
RESULTS_DIR = 'results'

# Cargar los mejores modelos
try:
    regression_model = joblib.load(os.path.join(MODELS_DIR, 'Random_Forest_Regressor_reg.pkl'))
    classification_model = joblib.load(os.path.join(MODELS_DIR, 'Gradient_Boosting_Classifier_clf.pkl'))
    
    # Cargar los datos preprocesados para obtener información sobre las características
    preprocessed_data = np.load('preprocessed_data.npz')
    X_train = preprocessed_data['X_train']
    feature_names = ['Edad', 'IMC', 'Presion_Sistolica', 'Presion_Diastolica', 'Perimetro_Abdominal', 
                    'Ejercicio', 'Consumo_Frutas', 'Cigarrillos', 'Diabetes_Familiar', 'Puntaje_Riesgo']
    
    print("Modelos cargados exitosamente")
except Exception as e:
    print(f"Error cargando modelos: {e}")
    regression_model = None
    classification_model = None

# Cargar resultados de comparación
try:
    with open(os.path.join(RESULTS_DIR, 'model_comparison.json'), 'r') as f:
        model_comparison = json.load(f)
    
    with open(os.path.join(RESULTS_DIR, 'ml_results.json'), 'r') as f:
        ml_results = json.load(f)
except Exception as e:
    print(f"Error cargando resultados: {e}")
    model_comparison = {}
    ml_results = {}

@app.route('/')
def index():
    """Página principal con información del proyecto"""
    return render_template('index.html', 
                         regression_available=regression_model is not None,
                         classification_available=classification_model is not None)

@app.route('/predict')
def predict_form():
    """Formulario para hacer predicciones"""
    return render_template('predict.html',
                         regression_available=regression_model is not None,
                         classification_available=classification_model is not None)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API para hacer predicciones"""
    try:
        data = request.get_json()
        
        # Extraer características del formulario
        features = [
            float(data.get('edad', 0)),
            float(data.get('imc', 0)),
            float(data.get('presion_sistolica', 0)),
            float(data.get('presion_diastolica', 0)),
            float(data.get('perimetro_abdominal', 0)),
            float(data.get('ejercicio', 0)),
            float(data.get('consumo_frutas', 0)),
            float(data.get('cigarrillos', 0)),
            float(data.get('diabetes_familiar', 0)),
            float(data.get('puntaje_riesgo', 0))
        ]
        
        # Convertir a array numpy
        X = np.array([features])
        
        results = {}
        
        # Predicción de regresión (nivel de glucosa)
        if regression_model is not None:
            glucose_prediction = regression_model.predict(X)[0]
            results['glucose_level'] = round(glucose_prediction, 2)
        
        # Predicción de clasificación (clase de diabetes)
        if classification_model is not None:
            class_prediction = classification_model.predict(X)[0]
            class_proba = classification_model.predict_proba(X)[0]
            
            class_names = ['Normal', 'Prediabetes', 'Diabetes']
            results['diabetes_class'] = class_names[class_prediction]
            results['class_probabilities'] = {
                class_names[i]: round(prob * 100, 2) 
                for i, prob in enumerate(class_proba)
            }
        
        return jsonify({
            'success': True,
            'predictions': results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/results')
def results():
    """Página con resultados y métricas de los modelos"""
    return render_template('results.html', 
                         model_comparison=model_comparison,
                         ml_results=ml_results)

@app.route('/api/results_data')
def api_results_data():
    """API para obtener datos de resultados"""
    try:
        return jsonify({
            'model_comparison': model_comparison,
            'ml_results': ml_results
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/visualizations')
def visualizations():
    """Página con visualizaciones de los resultados"""
    return render_template('visualizations.html')

@app.route('/api/generate_chart/<chart_type>')
def generate_chart(chart_type):
    """Genera gráficos dinámicos con Plotly"""
    try:
        if chart_type == 'regression_mse':
            # Gráfico de comparación MSE
            if 'regression' in model_comparison:
                models = list(model_comparison['regression']['MSE'].keys())
                mse_values = list(model_comparison['regression']['MSE'].values())
                
                fig = go.Figure(data=[
                    go.Bar(x=models, y=mse_values, 
                          marker_color='lightblue')
                ])
                fig.update_layout(
                    title='Comparación de MSE en Modelos de Regresión',
                    xaxis_title='Modelos',
                    yaxis_title='MSE'
                )
                
                return jsonify(json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig)))
        
        elif chart_type == 'classification_accuracy':
            # Gráfico de comparación de Accuracy
            if 'classification' in model_comparison:
                models = list(model_comparison['classification']['Accuracy'].keys())
                accuracy_values = list(model_comparison['classification']['Accuracy'].values())
                
                fig = go.Figure(data=[
                    go.Bar(x=models, y=accuracy_values,
                          marker_color='lightgreen')
                ])
                fig.update_layout(
                    title='Comparación de Accuracy en Modelos de Clasificación',
                    xaxis_title='Modelos',
                    yaxis_title='Accuracy'
                )
                
                return jsonify(json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig)))
        
        return jsonify({'error': 'Tipo de gráfico no encontrado'}), 404
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/about')
def about():
    """Página con información sobre el proyecto"""
    return render_template('about.html')

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)