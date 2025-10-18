# 2025/P Aplicaciones de Inteligencia Artificial (TI2082/D-IEI-N8-P1-C2/D Valdivia IEI)

## Proyecto: Predicción de Costos Médicos y Riesgo de Diabetes
### Descripción General
#### El objetivo de este proyecto es integrar dos modelos de Machine Learning, de regresión lineal para predecir el costo del seguro médico, y otro de clasificación para predecir la probabilidad de diabetes.
https://ev2-aplicacionesia.onrender.com/

### Tecnologías Utilizadas
####  - Lenguaje: Python 3.10+
####  - Framework Web: Flask
####  - Frontend: HTML5, Bootstrap 5, CSS3
### Librerías principales:
####  - scikit-learn (modelos ML)
####  - pandas, numpy (procesamiento de datos)
####  - matplotlib (visualización)
####  - joblib (serialización de modelos)

### Ejecución del Proyecto
### 1. Clonar el repositorio
#### git clone https://github.com/nlxntz/Ev_AIA.git
#### cd Ev2_AIA

### 2. Crear entorno virtual
#### python -m venv venv (ó  Ctrl + Shift + P en VSCode)
#### venv\Script\activate

### 3. Instalar dependencias
#### pip install -r requirements.txt

### 4. Ejecutar la aplicación
#### python run.py

### Modelos Implementados
### 1. Predicción de Diabetes
#### Tipo de modelo: Clasificación binaria (regresión Logística y Random Forest)
#### Objetivo: Determinar la probabilidad de que un paciente tenga diabates a partir de variables como: Glucosa, presión arterial, IMC, edad, etc.

### 2. Costo de Seguro Médico
#### Tipo de modelo: Regresión (Regresión lineal y Random Forest)
#### Objetivo: Estimar el costo del seguro médico de un paciente a partir de características como: Edad, sexo, fumador, región, BMI, hijos, etc.

### Analisis Solicitado
### 1) ¿Cuál es el umbral ideal para el modelo de predicción de diabetes?
#### El modelo de Regresión Logística entrega una probabilidad entre 0 y 1.
#### Por defecto, el umbral es 0.5, sin embargo, tras evaluar las métricas (precisión, recall y F1-score), se determinó que el umbral óptimo es 0.38, el cual mejora la sensibilidad (detección de casos positivos) sin perder demasiada precisión.
### 2) ¿Cuales son los factores que más influyen en el precio de los costos asociados al seguro médico?
####
