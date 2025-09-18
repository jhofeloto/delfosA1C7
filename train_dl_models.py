import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import json

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

# Función para crear modelo de regresión
def create_reg_model(input_shape):
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Función para crear modelo de clasificación
def create_clf_model(input_shape, num_classes):
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Entrenar modelo de regresión
reg_model = create_reg_model(X_train.shape[1])
history_reg = reg_model.fit(X_train, y_reg_train, validation_data=(X_val, y_reg_val), epochs=100, batch_size=16, verbose=0)
reg_model.save('models/dl_reg_model.h5')

# Evaluar regresión
y_pred_reg = reg_model.predict(X_test).flatten()
reg_metrics = {
    'MSE': mean_squared_error(y_reg_test, y_pred_reg),
    'MAE': mean_absolute_error(y_reg_test, y_pred_reg),
    'R2': r2_score(y_reg_test, y_pred_reg)
}

# Entrenar modelo de clasificación
num_classes = len(np.unique(y_clf_train))
clf_model = create_clf_model(X_train.shape[1], num_classes)
history_clf = clf_model.fit(X_train, y_clf_train, validation_data=(X_val, y_clf_val), epochs=100, batch_size=16, verbose=0)
clf_model.save('models/dl_clf_model.h5')

# Evaluar clasificación
y_pred_clf = np.argmax(clf_model.predict(X_test), axis=1)
y_proba_clf = clf_model.predict(X_test)

acc = accuracy_score(y_clf_test, y_pred_clf)
prec = precision_score(y_clf_test, y_pred_clf, average='weighted', zero_division=0)
rec = recall_score(y_clf_test, y_pred_clf, average='weighted', zero_division=0)
f1 = f1_score(y_clf_test, y_pred_clf, average='weighted', zero_division=0)
auc = None
if len(np.unique(y_clf_test)) > 1:
    try:
        auc = roc_auc_score(y_clf_test, y_proba_clf, multi_class='ovr')
    except ValueError:
        auc = None
cm = confusion_matrix(y_clf_test, y_pred_clf).tolist()

clf_metrics = {
    'Accuracy': acc,
    'Precision': prec,
    'Recall': rec,
    'F1': f1,
    'AUC': auc,
    'Confusion Matrix': cm
}

# Guardar resultados
dl_results = {'regression': reg_metrics, 'classification': clf_metrics}
with open('results/dl_results.json', 'w') as f:
    json.dump(dl_results, f, indent=4)

print("Modelos DL entrenados y evaluados. Resultados guardados en results/dl_results.json")