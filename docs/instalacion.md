# Instalación

Probado con Python 3.13 en Windows, sin GPU. Ejecutar todo desde la raíz del repositorio.

## 1. Entorno

```powershell
python -m venv .venv
.venv\Scripts\activate        # Windows (Linux/macOS: source .venv/bin/activate)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # PyTorch CPU
pip install -r requirements.txt
```

PyTorch se instala primero para que `pip` conserve la versión CPU al leer `requirements.txt`.

## 2. Datos

Las imágenes procesadas, las máscaras y las características ya están versionadas. Solo hace falta
descargar Kaggle para regenerarlas desde cero: ver `data/README.md`.

## 3. Modelos entrenados

Los modelos (`experiments/checkpoints/`) no se versionan. Para generar los clásicos:

```powershell
$env:PYTHONUTF8 = "1"
python -m nbconvert --to notebook --execute --inplace notebooks/04_modeling_ml_clasico.ipynb
```

Entre otros, produce `ml_mejor.joblib` (SVM con probabilidades, para la app) y
`ml_mejor_imagen_completa.joblib` (experimento del atajo). Hay que regenerarlos si cambia la versión de
scikit-learn: un `.joblib` no es compatible entre versiones.

## 4. Aplicación (pendiente)

```powershell
streamlit run app/app.py
```
