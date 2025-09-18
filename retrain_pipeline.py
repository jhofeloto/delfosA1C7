import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

class RetrainPipeline:
    def __init__(self, original_data_path='data/output-glucosa_labeled.csv', models_dir='models'):
        self.original_data_path = original_data_path
        self.models_dir = models_dir
        self.label_encoders = {}
        self.scaler = None
        self.feature_columns = None

    def load_original_data(self):
        """Cargar datos originales"""
        df = pd.read_csv(self.original_data_path)
        # Eliminar columnas irrelevantes
        columns_to_drop = ['identificacion', 'Fecha_Fin_Registro', 'ips_codigo', 'nombres', 'apellidos', 'tipo_identificacion',
                           'fecha_nacimiento', 'telefono', 'Municipio', 'direccion', 'Nombre_Completo', 'Examen', 'Fecha_Examen',
                           'Grupo_Analito', 'Analito', 'Regimen', 'responsable_registro', 'servicio', 'Edad_Años']
        df = df.drop(columns=columns_to_drop, errors='ignore')
        return df

    def add_new_data(self, new_data_path):
        """Añadir nuevos datos al dataset"""
        original_df = self.load_original_data()
        new_df = pd.read_csv(new_data_path)
        # Aplicar mismo preprocesamiento
        new_df = new_df.drop(columns=['identificacion', 'Fecha_Fin_Registro', 'ips_codigo', 'nombres', 'apellidos', 'tipo_identificacion',
                                      'fecha_nacimiento', 'telefono', 'Municipio', 'direccion', 'Nombre_Completo', 'Examen', 'Fecha_Examen',
                                      'Grupo_Analito', 'Analito', 'Regimen', 'responsable_registro', 'servicio', 'Edad_Años'], errors='ignore')
        combined_df = pd.concat([original_df, new_df], ignore_index=True)
        combined_df = combined_df.dropna()
        return combined_df

    def preprocess_data(self, df):
        """Preprocesar datos"""
        # Encoding categórico
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            df[col] = self.label_encoders[col].fit_transform(df[col])

        # Asegurar Clase_DM como int
        df['Clase_DM'] = df['Clase_DM'].astype(int)

        # Escalar numéricos
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = numeric_cols.drop(['Resultado'])
        if self.scaler is None:
            self.scaler = StandardScaler()
        df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])

        # Guardar columnas de características
        if self.feature_columns is None:
            self.feature_columns = df.drop(['Resultado', 'Clase_DM'], axis=1).columns.tolist()

        return df

    def retrain_models(self, new_data_path=None):
        """Reentrenar modelos con nuevos datos"""
        if new_data_path:
            df = self.add_new_data(new_data_path)
        else:
            df = self.load_original_data()

        df = self.preprocess_data(df)

        # Separar features y targets
        X = df[self.feature_columns]
        y_reg = df['Resultado']
        y_clf = df['Clase_DM']

        # Dividir datos
        X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
            X, y_reg, y_clf, test_size=0.2, random_state=42)

        # Reentrenar modelos ML
        from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
        from sklearn.svm import SVR, SVC

        # Regresión
        reg_models = {
            'Linear_Regression': LinearRegression(),
            'Ridge_Regression': Ridge(),
            'Random_Forest_Regressor': RandomForestRegressor(random_state=42),
            'Gradient_Boosting_Regressor': GradientBoostingRegressor(random_state=42)
        }

        for name, model in reg_models.items():
            model.fit(X_train, y_reg_train)
            joblib.dump(model, os.path.join(self.models_dir, f'{name}_reg.pkl'))

        # Clasificación
        clf_models = {
            'Logistic_Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random_Forest_Classifier': RandomForestClassifier(random_state=42),
            'SVM_Classifier': SVC(random_state=42, probability=True),
            'Gradient_Boosting_Classifier': GradientBoostingClassifier(random_state=42)
        }

        for name, model in clf_models.items():
            model.fit(X_train, y_clf_train)
            joblib.dump(model, os.path.join(self.models_dir, f'{name}_clf.pkl'))

        print("Modelos reentrenados exitosamente con nuevos datos.")

# Ejemplo de uso
if __name__ == "__main__":
    pipeline = RetrainPipeline()
    # Para reentrenar con datos originales
    pipeline.retrain_models()
    # Para añadir nuevos datos: pipeline.retrain_models('path/to/new_data.csv')