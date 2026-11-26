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

Para entrenar la CNN y sus experimentos (~6 h en CPU; omite los que ya estén terminados):

```powershell
python -m src.training.entrenar_cnn
python -m nbconvert --to notebook --execute --inplace notebooks/04_modeling_cnn.ipynb
```

El notebook también usa tres redes de una ablación repetida (`sin_aumento_s43` y `sin_aumento_completa_s42`, además de
`sin_aumento`). Las dos primeras no las lanza el script: sus métricas están versionadas en `experiments/results/cnn/`.

## 4. Aplicación

Requiere `experiments/checkpoints/ml_mejor.joblib` (sección 3). La CNN aparece como segunda opción solo si existe
`experiments/checkpoints/cnn_filtros_x2.pt`.

```powershell
.venv\Scripts\activate
streamlit run app/app.py
```

Se abre en el navegador (http://localhost:8501). El navegador pide permiso para usar la cámara; también se puede
subir una imagen (JPG, PNG o HEIC).

Uso:

1. Elegir la fruta en la barra lateral (los cortes de tamaño son distintos por fruta) y el modelo (SVM por defecto,
   el recomendado en el notebook 05).
2. Tomar la foto: una fruta centrada, fondo liso, luz pareja y **siempre a la misma distancia**.
3. La app muestra la máscara de la fruta (para comprobar la segmentación), la calidad con sus probabilidades y el
   tamaño: `d_eq = 2·√(área de la máscara / π)` y su tercil por fruta, con los cortes de `train`.

El preprocesamiento es el mismo del entrenamiento (`src/evaluation/prediccion.py`): EXIF, cuadrar con relleno,
224×224, GrabCut y el modelo. Con el SVM la clase sale de `predict`; si no coincide con la probabilidad mayor, la
app lo advierte. También avisa si la máscara parece mala (vacía, demasiado grande o tocando el borde).
