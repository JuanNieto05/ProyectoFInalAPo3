# Guía para la CNN, la evaluación final y la app

Para quien siga con los pasos técnicos que faltan. Complementa a [GUIA_SIGUIENTE_ETAPA.md](GUIA_SIGUIENTE_ETAPA.md)
(datos), que sigue vigente: **leer sus reglas de la sección 2 antes de empezar**.

---

## 1. Qué hay hecho

| Paso | Dónde | Resultado clave |
|---|---|---|
| EDA | `notebooks/02_data_understanding_eda.ipynb` | El **fondo delata la clase** en Kaggle (un modelo que solo ve el borde logra F1 ≈ 0,89 en good/bad). Fotos propias: más oscuras, fruta más pequeña en la imagen. |
| Segmentación | `notebooks/03_data_preparation.ipynb`, `src/data/segmentacion.py`, `data/masks/` | GrabCut con priors aísla la fruta y **elimina el atajo**. Tamaño = `d_eq` (diámetro equivalente de la máscara). |
| ML clásico | `notebooks/04_modeling_ml_clasico.ipynb`, `src/models/clasicos.py`, `src/features/` | 80 características sobre la fruta. **Mejor: SVM RBF (C=30, γ=0,003), macro-F1 0,727 en val** (F1 good 0,90, regular 0,33, bad 0,94). Línea base 0,22–0,31. |

Módulos reutilizables:

```python
from src.data.carga import cargar_splits, cargar_imagen, cargar_mascara   # datos (siempre desde splits.csv)
from src.data.segmentacion import segmentar_fruta, aplicar_mascara        # segmentar una imagen nueva / pintar fondo gris
from src.features.extraccion import extraer, NOMBRES                       # 80 características (para el SVM)
from src.evaluation.metricas import resumen, tabla, intervalo_bootstrap    # métricas del proyecto
```

## 2. Preparar el entorno

```powershell
pip install -r requirements.txt          # incluye opencv, scikit-image, xgboost, pillow-heif
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # PyTorch CPU (no hay GPU)
```

Los modelos entrenados (`experiments/checkpoints/*.joblib`) **no están en git**. Para tenerlos, ejecutar
`notebooks/04_modeling_ml_clasico.ipynb` completo (~10 min). Genera `ml_mejor.joblib` (SVM con probabilidades)
y `ml_mejor_imagen_completa.joblib` (el mismo SVM entrenado con fondo, para el experimento del atajo).

---

## 3. CNN (notebook `04_modeling_cnn.ipynb`)

Requisito del enunciado: CNN pequeña (2–3 capas conv + pooling + densas), de preferencia **desde cero**.

Recomendaciones según lo que encontramos:

- **Entrada: la fruta segmentada** (`aplicar_mascara(rgb, cargar_mascara(fila))`, fondo gris). Si se entrena
  con la imagen completa, la CNN aprenderá el fondo. Opcional: entrenar también una versión con imagen
  completa para el experimento del atajo (igual que con el SVM).
- **Tamaño 128×128** (se entrena en CPU: ~1 min por época; 224×224 es ~3 veces más lento).
- **Desbalance:** `CrossEntropyLoss(weight=...)` con pesos `n / (K · n_c)` calculados en train
  (good 0,65, regular 8,87, bad 0,75), o un `WeightedRandomSampler`. Sin esto la CNN nunca predice "regular"
  (al SVM le pasó: recall 0).
- **Aumento de datos solo en train:** giros, rotaciones, recortes, y **variaciones de brillo/color** (las fotos
  propias son más oscuras).
- **Normalización:** media/desviación por canal calculadas **solo con train**.
- **Selección:** early stopping por **macro-F1 en val** (no por accuracy ni por loss). Guardar curvas de
  loss y macro-F1 de train/val por época (para el análisis de sobreajuste del informe).
- Fijar semillas (`torch.manual_seed`, `numpy`, `random`) y guardar el mejor modelo en `experiments/checkpoints/`.

## 4. Evaluación final (notebook `05_evaluation.ipynb`)

- Evaluar **una sola vez** los modelos ya elegidos en `test` y `test_propias`: línea base, SVM y CNN.
- **Por separado** en `test` (Kaggle) y `test_propias` (fotos propias). La diferencia mide el cambio de dominio.
- En `test_propias` (46 imágenes) reportar **intervalos bootstrap** (`intervalo_bootstrap`), no solo la cifra.
- **Experimento del atajo:** SVM (y CNN) entrenados con imagen completa vs con fruta segmentada, comparados en
  `test_propias`. En `val` rindieron igual (0,73); la hipótesis es que con fondos nuevos la versión con fruta
  segmentada generaliza mejor. Sea cual sea el resultado, es un hallazgo para el informe.
- Matrices de confusión, métricas por fruta, y **análisis de errores con imágenes** (¿regular se confunde con good o con bad?).
- Figuras en `docs/figures/` en SVG; métricas en `experiments/results/`.

## 5. App (carpeta `app/`, Streamlit)

- Cargar imagen o capturar con `st.camera_input`.
- **Mismo preprocesamiento que en el entrenamiento**, en este orden:
  1. corregir rotación EXIF y pasar a RGB (`load_image` en `data/prepare_dataset.py`);
  2. cuadrar con relleno del color de fondo y redimensionar a 224×224 (`pad_square_resize`);
  3. `segmentar_fruta` → máscara;
  4. SVM: `extraer(rgb, mascara)` → `ml_mejor.joblib`; CNN: `aplicar_mascara`, redimensionar y normalizar igual que en train.
- **SVM: la clase se decide con `predict`, no con el máximo de `predict_proba`.** Verificado en val: `predict` da
  macro-F1 0,727 (recall de regular 0,53); tomar la clase más probable da 0,630 y **nunca** predice regular (recall 0),
  probablemente porque la calibración de Platt no conserva los pesos de clase. Difieren en 74 de 917 imágenes: en esos casos la
  probabilidad mostrada no respalda la clase, y la app debe tenerlo en cuenta.
- Mostrar: clase de calidad, confianza (probabilidades), **tamaño** (`d_eq = 2·sqrt(área_máscara/π)` y
  pequeño/mediano/grande) y la máscara superpuesta (ayuda a ver si la segmentación falló).
- El tamaño en píxeles **depende de la distancia de la cámara**: decir en la app que la foto debe tomarse a una
  distancia fija (y en el informe, que los cortes deben calibrarse para ese montaje).

## 6. Lista de verificación

- [ ] CNN desde cero sobre la fruta segmentada, con pesos de clase y aumento de datos
- [ ] Curvas de entrenamiento (loss y macro-F1, train y val)
- [ ] Evaluación única en `test` y `test_propias`, con intervalos bootstrap en propias
- [ ] Experimento del atajo (imagen completa vs fruta) en `test_propias`
- [ ] Análisis de errores con imágenes
- [ ] App con carga de imagen y cámara, mismo preprocesamiento, calidad + confianza + tamaño
- [ ] `requirements.txt` actualizado (torch, torchvision) y `docs/instalacion.md` con cómo lanzar la app
