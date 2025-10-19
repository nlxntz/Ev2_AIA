# 2025/P Aplicaciones de Inteligencia Artificial (TI2082/D-IEI-N8-P1-C2/D Valdivia IEI)

## Proyecto: Predicción de Costos Médicos y Riesgo de Diabetes
### Descripción General
#### El objetivo de este proyecto es integrar dos modelos de Machine Learning, de regresión lineal para predecir el costo del seguro médico, y otro de clasificación para predecir la probabilidad de diabetes.
https://ev2-aplicacionesia.onrender.com/

## Tecnologías Utilizadas
####  - Lenguaje: Python 3.10+
####  - Framework Web: Flask
####  - Frontend: HTML5, Bootstrap 5, CSS3
## Librerías principales:
####  - scikit-learn (modelos ML)
####  - pandas, numpy (procesamiento de datos)
####  - matplotlib (visualización)
####  - joblib (serialización de modelos)

## Ejecución del Proyecto
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

## Modelos Implementados
### 1. Predicción de Diabetes
#### Tipo de modelo: Clasificación binaria (regresión Logística y Random Forest)
#### Objetivo: Determinar la probabilidad de que un paciente tenga diabates a partir de variables como: Glucosa, presión arterial, IMC, edad, etc.

### 2. Costo de Seguro Médico
#### Tipo de modelo: Regresión (Regresión lineal y Random Forest)
#### Objetivo: Estimar el costo del seguro médico de un paciente a partir de características como: Edad, sexo, fumador, región, BMI, hijos, etc.

## Analisis Solicitado
### 1) ¿Cuál es el umbral ideal para el modelo de predicción de diabetes?
#### El umbral ideal se determinó analizando la relación entre sensibilidad y especificidad.
#### Un valor cercano a 0.4 ofreció el mejor equilibrio, priorizando la detección temprana de casos positivos, sin aumentar excesivamente los falsos positivos.
#### Esto es importante en contextos médicos, donde es preferible detectar posibles pacientes antes que omitir un caso real.
### 2) ¿Cuales son los factores que más influyen en el precio de los costos asociados al seguro médico?
#### Según la importancia de variables del modelo de regresión, los factores más influyentes fueron:
#### - Edad: a mayor edad, mayor costo esperado.
#### - IMC: altos valores se asocian con mayor riesgo de enfermedades.
#### - Ser fumador: el factor con mayor impacto, incrementa significativamente el costo del seguro.
#### - Número de hijos: influye moderadamente, al igual que la región geográfica.
### 3) Análisis comparativo de las características de ambos modelos con RandomForest
#### El modelo Random Forest mostró mejor rendimiento frente a modelos más simples (como regresión logística o lineal) gracias a su capacidad de capturar relaciones no lineales.
#### En ambos casos:
#### - Para diabetes, RandomForest mejoró la precisión y el recall respecto a la regresión logística.
#### - Para seguros médicos, redujo el error cuadrático medio y mejoró la generalización frente a la regresión lineal.
### 4) ¿Qué técnica de optimización mejora el rendimiento de ambos modelos?
#### - El número de estimadores (n_estimators)
#### - La profundidad máxima (max_depth)
#### - Y los criterios de división (criterion).
### 5) Contexto de los datos
#### - Dataset de diabetes: proviene de un estudio clínico sobre factores que influyen en la aparición de diabetes tipo II. Incluye variables como glucosa, presión arterial, edad y número de embarazos.
#### - Dataset de seguros médicos: contiene información demográfica y de salud de personas, junto a los costos asociados a sus seguros.
### 6) Análisis del sesgo de los modelos
#### Ambos modelos presentan cierto sesgo hacia las clases o valores más frecuentes:
#### - En el modelo de diabetes, la clase “no diabético” es predominante, lo que genera un sesgo hacia la predicción negativa (subrepresentación de casos positivos).
#### - En el modelo de seguros, el sesgo surge por la distribución desigual de fumadores y edades extremas, lo que afecta la predicción en grupos minoritarios.
