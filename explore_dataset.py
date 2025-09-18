import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el dataset
df = pd.read_csv('data/output-glucosa_labeled.csv')

# Información básica
print("Tamaño del dataset:", df.shape)
print("\nTipos de datos:")
print(df.dtypes)
print("\nValores faltantes por columna:")
print(df.isnull().sum())
print("\nEstadísticas descriptivas:")
print(df.describe())

# Distribución de clases
print("\nDistribución de Clase_DM:")
print(df['Clase_DM'].value_counts())

# Visualizaciones básicas
plt.figure(figsize=(10, 6))
sns.histplot(df['Resultado'], bins=20, kde=True)
plt.title('Distribución de Glucosa (Resultado)')
plt.xlabel('Glucosa (mg/dL)')
plt.ylabel('Frecuencia')
plt.savefig('glucosa_distribution.png')
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(x='Clase_DM', data=df)
plt.title('Distribución de Clases de Diabetes')
plt.savefig('clase_dm_distribution.png')
plt.show()

# Correlación con variables numéricas
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlation = df[numeric_cols].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Matriz de Correlación')
plt.savefig('correlation_matrix.png')
plt.show()