import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Cargar el dataset
df = pd.read_csv('data/output-glucosa_labeled.csv')

# Eliminar columnas irrelevantes (identificadores, nombres, etc.)
columns_to_drop = ['identificacion', 'Fecha_Fin_Registro', 'ips_codigo', 'nombres', 'apellidos', 'tipo_identificacion',
                   'fecha_nacimiento', 'telefono', 'Municipio', 'direccion', 'Nombre_Completo', 'Examen', 'Fecha_Examen',
                   'Grupo_Analito', 'Analito', 'Regimen', 'responsable_registro', 'servicio', 'Edad_Años']
df = df.drop(columns=columns_to_drop, errors='ignore')

# Manejar valores faltantes
df = df.dropna()  # Para simplicidad, eliminar filas con NaN

# Encoding de variables categóricas
categorical_cols = df.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Asegurar que Clase_DM sea entero
df['Clase_DM'] = df['Clase_DM'].astype(int)

# Escalar características numéricas
numeric_cols = df.select_dtypes(include=[np.number]).columns
numeric_cols = numeric_cols.drop(['Resultado'])  # No escalar el target de regresión
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Separar features y targets
X = df.drop(['Resultado', 'Clase_DM'], axis=1)
y_reg = df['Resultado']
y_clf = df['Clase_DM']

# Dividir en train/val/test (70/15/15)
X_train, X_temp, y_reg_train, y_reg_temp = train_test_split(X, y_reg, test_size=0.3, random_state=42)
X_val, X_test, y_reg_val, y_reg_test = train_test_split(X_temp, y_reg_temp, test_size=0.5, random_state=42)

X_train_clf, X_temp_clf, y_clf_train, y_clf_temp = train_test_split(X, y_clf, test_size=0.3, random_state=42)
X_val_clf, X_test_clf, y_clf_val, y_clf_test = train_test_split(X_temp_clf, y_clf_temp, test_size=0.5, random_state=42)

# Guardar los datos preprocesados
np.savez('preprocessed_data.npz',
         X_train=X_train, X_val=X_val, X_test=X_test,
         y_reg_train=y_reg_train, y_reg_val=y_reg_val, y_reg_test=y_reg_test,
         y_clf_train=y_clf_train, y_clf_val=y_clf_val, y_clf_test=y_clf_test)

print("Datos preprocesados guardados en preprocessed_data.npz")
print("Tamaño de entrenamiento:", X_train.shape)
print("Tamaño de validación:", X_val.shape)
print("Tamaño de prueba:", X_test.shape)