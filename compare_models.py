import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Cargar resultados ML
with open('results/ml_results.json', 'r') as f:
    ml_results = json.load(f)

# Cargar resultados DL (si existe)
dl_results = {}
try:
    with open('results/dl_results.json', 'r') as f:
        dl_results = json.load(f)
except FileNotFoundError:
    print("Resultados DL no encontrados, omitiendo.")

# Función para extraer métricas de test
def extract_test_metrics(results, task):
    test_metrics = {}
    for model, metrics in results[task].items():
        test_metrics[model] = metrics['Test']
    return test_metrics

# Extraer métricas de regresión
reg_metrics = extract_test_metrics(ml_results, 'regression')
if dl_results:
    reg_metrics['Deep Learning Regressor'] = dl_results['regression']

# Extraer métricas de clasificación
clf_metrics = extract_test_metrics(ml_results, 'classification')
if dl_results:
    clf_metrics['Deep Learning Classifier'] = dl_results['classification']

# Crear DataFrames para comparación
reg_df = pd.DataFrame(reg_metrics).T
clf_df = pd.DataFrame(clf_metrics).T

print("Métricas de Regresión (Test):")
print(reg_df)
print("\nMétricas de Clasificación (Test):")
print(clf_df)

# Visualizaciones
# Regresión: Comparación de MSE
plt.figure(figsize=(10, 6))
reg_df['MSE'].plot(kind='bar')
plt.title('Comparación de MSE en Regresión')
plt.ylabel('MSE')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('results/regression_mse_comparison.png')
plt.show()

# Clasificación: Comparación de Accuracy
plt.figure(figsize=(10, 6))
clf_df['Accuracy'].plot(kind='bar')
plt.title('Comparación de Accuracy en Clasificación')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('results/classification_accuracy_comparison.png')
plt.show()

# Matriz de confusión para el mejor modelo de clasificación
best_clf_model = clf_df['Accuracy'].idxmax()
best_clf_results = ml_results['classification'][best_clf_model]['Test']
cm = best_clf_results['Confusion Matrix']

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Matriz de Confusión - {best_clf_model}')
plt.xlabel('Predicho')
plt.ylabel('Real')
plt.savefig('results/best_clf_confusion_matrix.png')
plt.show()

# Seleccionar mejores modelos
best_reg = reg_df['R2'].idxmax()
best_clf = clf_df['Accuracy'].idxmax()

print(f"\nMejor modelo de regresión: {best_reg} (R² = {reg_df.loc[best_reg, 'R2']:.4f})")
print(f"Mejor modelo de clasificación: {best_clf} (Accuracy = {clf_df.loc[best_clf, 'Accuracy']:.4f})")

# Guardar comparación
comparison = {
    'regression': reg_df.to_dict(),
    'classification': clf_df.to_dict(),
    'best_models': {'regression': best_reg, 'classification': best_clf}
}
with open('results/model_comparison.json', 'w') as f:
    json.dump(comparison, f, indent=4)

print("Comparación completada. Resultados guardados en results/model_comparison.json")