import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib
import pandas as pd

# Cargar datos preprocesados
data = np.load('preprocessed_data.npz')
X_train = data['X_train']
X_val = data['X_val']
X_test = data['X_test']
y_reg_train = data['y_reg_train']
y_reg_val = data['y_reg_val']
y_reg_test = data['y_reg_test']
y_clf_train = data['y_clf_train'].astype(int)
y_clf_val = data['y_clf_val'].astype(int)
y_clf_test = data['y_clf_test'].astype(int)

# Modelos de regresión
reg_models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Random Forest Regressor': RandomForestRegressor(random_state=42),
    'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=42)
}

# Modelos de clasificación
clf_models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest Classifier': RandomForestClassifier(random_state=42),
    'SVM Classifier': SVC(random_state=42, probability=True),
    'Gradient Boosting Classifier': GradientBoostingClassifier(random_state=42)
}

# Función para evaluar regresión
def evaluate_regression(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'MSE': mse, 'MAE': mae, 'R2': r2}

# Función para evaluar clasificación
def evaluate_classification(y_true, y_pred, y_proba=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    auc = None
    if y_proba is not None and len(np.unique(y_true)) > 1:
        try:
            auc = roc_auc_score(y_true, y_proba, multi_class='ovr')
        except ValueError:
            auc = None
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1, 'AUC': auc, 'Confusion Matrix': cm}

# Entrenar y evaluar modelos de regresión
reg_results = {}
for name, model in reg_models.items():
    model.fit(X_train, y_reg_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)
    val_metrics = evaluate_regression(y_reg_val, y_pred_val)
    test_metrics = evaluate_regression(y_reg_test, y_pred_test)
    reg_results[name] = {'Validation': val_metrics, 'Test': test_metrics}
    joblib.dump(model, f'models/{name.replace(" ", "_")}_reg.pkl')

# Entrenar y evaluar modelos de clasificación
clf_results = {}
for name, model in clf_models.items():
    model.fit(X_train, y_clf_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)
    y_proba_val = model.predict_proba(X_val) if hasattr(model, 'predict_proba') else None
    y_proba_test = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    val_metrics = evaluate_classification(y_clf_val, y_pred_val, y_proba_val)
    test_metrics = evaluate_classification(y_clf_test, y_pred_test, y_proba_test)
    clf_results[name] = {'Validation': val_metrics, 'Test': test_metrics}
    joblib.dump(model, f'models/{name.replace(" ", "_")}_clf.pkl')

# Guardar resultados
import json
with open('results/ml_results.json', 'w') as f:
    json.dump({'regression': reg_results, 'classification': clf_results}, f, indent=4)

print("Modelos ML entrenados y evaluados. Resultados guardados en results/ml_results.json")