from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np
import os
import matplotlib.pyplot as plt
import io
import base64
import uvicorn
import pandas as pd

app = FastAPI(title="Predicción de Seguros Médicos y Diabetes")

# ================= RUTAS =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# ================= MODELOS =================
insurance_model = joblib.load(os.path.join(MODELS_DIR, "insurance_model.pkl"))
insurance_rf_model = joblib.load(os.path.join(MODELS_DIR, "insurance_rf_model.pkl"))
insurance_columns = joblib.load(os.path.join(MODELS_DIR, "insurance_columns.pkl"))
diabetes_model = joblib.load(os.path.join(MODELS_DIR, "diabetes_model.pkl"))
diabetes_rf_model = joblib.load(os.path.join(MODELS_DIR, "diabetes_rf_model.pkl"))

# ================= UTILS =================
def plot_feature_importance(model, feature_names):
    import matplotlib.pyplot as plt
    import io
    import base64

    # Mapeo de nombres a español
    column_map = {
        "age": "Edad",
        "bmi": "Índice de Masa Corporal",
        "children": "Número de hijos",
        "smoker_yes": "Fumador",
        "sex_male": "Sexo masculino",
        "region_northwest": "Región Noroeste",
        "region_southeast": "Región Sureste",
        "region_southwest": "Región Suroeste"
    }
    feature_names_es = [column_map.get(f, f) for f in feature_names]

    # Generar gráfico
    importances = model.feature_importances_
    plt.figure(figsize=(7,5))
    plt.barh(feature_names_es, importances, color='#4B9CD3')
    plt.title("📊 Importancia de Variables (Random Forest)")
    plt.xlabel("Importancia")
    plt.ylabel("Características")
    plt.tight_layout()

    # Guardar gráfico en base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img_base64

# Lista de regiones de Chile normalizada para one-hot
REGIONES_CHILE = [
    "arica y parinacota", "tarapaca", "antofagasta", "atacama", "coquimbo",
    "valparaiso", "metropolitana de santiago", "ohiggins", "maule", "ñuble",
    "biobio", "la araucania", "los rios", "los lagos", "aysen",
    "magallanes y de la antartica chilena"
]

def normalize_region(region):
    return region.replace(" ", "_").replace("ó","o").replace("í","i").lower()

# ================= RUTAS =================
@app.get("/ping")
def ping():
    return {"message": "Servidor funcionando ✅"}

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ================= DIABETES =================
@app.get("/diabetes", response_class=HTMLResponse)
def diabetes_form(request: Request):
    return templates.TemplateResponse("diabetes.html", {"request": request, "result": None})

@app.post("/predict_diabetes", response_class=HTMLResponse)
async def predict_diabetes(
    request: Request,
    pregnancies: float = Form(...),
    glucose: float = Form(...),
    bloodpressure: float = Form(...),
    skinthickness: float = Form(...),
    insulin: float = Form(...),
    bmi: float = Form(...),
    diabetespedigreefunction: float = Form(...),
    age: float = Form(...)
):
    features = np.array([[pregnancies, glucose, bloodpressure,
                          skinthickness, insulin, bmi,
                          diabetespedigreefunction, age]])
    
    # Logistic Regression
    prob_lr = diabetes_model.predict_proba(features)[0][1]
    pred_lr = int(prob_lr >= 0.47)  # Umbral ideal
    # Random Forest
    prob_rf = diabetes_rf_model.predict_proba(features)[0][1]
    pred_rf = int(prob_rf >= 0.5)
    
    result = {
        "prob_lr": round(prob_lr, 4),
        "pred_lr": pred_lr,
        "prob_rf": round(prob_rf, 4),
        "pred_rf": pred_rf
    }
    
    return templates.TemplateResponse("diabetes.html", {"request": request, "result": result})

# ================= SEGURO MÉDICO =================
@app.get("/insurance", response_class=HTMLResponse)
def insurance_form(request: Request):
    return templates.TemplateResponse("insurance.html", {"request": request, "result": None, "feature_plot": None, "regiones": REGIONES_CHILE})

@app.post("/predict_insurance", response_class=HTMLResponse)
async def predict_insurance(
    request: Request,
    age: float = Form(...),
    bmi: float = Form(...),
    children: int = Form(...),
    smoker: str = Form(...),
    sex: str = Form(...),
    region: str = Form(...)
):
    # Crear input básico
    df_input = pd.DataFrame({
        "age": [age],
        "bmi": [bmi],
        "children": [children],
        "smoker_yes": [1 if smoker=="Sí" else 0],
        "sex_male": [1 if sex=="Masculino" else 0]
    })

    # Convertir región a columnas dummy
    region_cols = [col for col in insurance_columns if col.startswith('region_')]
    region_input = {col: 0 for col in region_cols}
    key = f"region_{normalize_region(region)}"
    if key in region_input:
        region_input[key] = 1
    for col in region_input:
        df_input[col] = region_input[col]

    # Completar columnas faltantes
    for col in insurance_columns:
        if col not in df_input.columns:
            df_input[col] = 0
    df_input = df_input[insurance_columns]

    # Predicciones
    lr_pred = insurance_model.predict(df_input)[0]
    rf_pred = insurance_rf_model.predict(df_input)[0]

    # Gráfico importancia de variables
    feature_plot = plot_feature_importance(insurance_rf_model, insurance_columns)

    # Resultados en español
    result = {
        "Costo estimado (Regresión Lineal)": round(lr_pred,2),
        "Costo estimado (Random Forest)": round(rf_pred,2)
    }

    return templates.TemplateResponse(
        "insurance.html", 
        {"request": request, "result": result, "feature_plot": feature_plot, "regiones": REGIONES_CHILE}
    )

# ================= RUN =================
if __name__ == "__main__":
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)