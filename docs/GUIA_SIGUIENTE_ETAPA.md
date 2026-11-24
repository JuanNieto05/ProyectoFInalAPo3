# Guía para continuar el proyecto (después de la fase de datos)

Esta guía es para quien siga con el modelado, la evaluación y el despliegue.
Resume qué hay, qué NO se debe tocar y qué falta según el enunciado.

> **Actualización (rama `fix/limpieza-datos`).** Después de la primera versión de esta guía se corrigieron
> tres problemas de datos: (1) una foto propia de naranja estaba a la vez en `good` y `regular`;
> (2) quedaban 31 parejas de fotos casi idénticas con etiquetas distintas; (3) el split por imagen dejaba
> tomas casi iguales de la misma fruta repartidas entre train y test (fuga). Ver la sección 4.
> Los conteos de abajo ya son los nuevos.

---

## 1. Qué se entrega de la fase de datos

| Archivo / carpeta | Qué es |
|---|---|
| `data/processed/{good,regular,bad}/` | Imágenes finales, 224×224 RGB, fondo rellenado (sin deformar). La carpeta es la clase de calidad. Versionadas en git. |
| `data/manifest.csv` | Una fila por imagen: fruta, calidad, fuente (`own`/`kaggle`), archivo original, `area_ratio`, `diam_ratio`, `relabeled_from`. |
| `data/splits.csv` | **Fuente única de verdad de los splits** + columnas `grupo`, `tamano`, `d_eq` y `mask_area`. Todos deben cargar los datos desde aquí. |
| `data/masks/` + `data/masks.csv` | Máscara de la fruta (GrabCut) de cada imagen. Cargar con `src.data.carga.cargar_mascara`. |
| `data/duplicates.csv` | Parejas de duplicados detectadas (evidencia de la limpieza). |
| `data/conflicts_removed.csv` | Las 31 imágenes eliminadas por conflicto de etiqueta y su gemela conservada. |
| `data/fotos_originales/` | Fotos propias originales (respaldo). |

Las carpetas `data/duplicates_removed/` y `data/conflicts_removed/` (imágenes retiradas) **no** se versionan.

Tres frutas (apple, banana, orange) × tres calidades (good, regular, bad). Total: **5.705 imágenes**.

| split | good | regular | bad | total | uso |
|---|---|---|---|---|---|
| train | 2002 | 146 | 1736 | 3884 | entrenar |
| val | 440 | 32 | 445 | 917 | hiperparámetros y selección de modelo |
| test | 424 | 32 | 402 | 858 | evaluación final (Kaggle) |
| test_propias | 15 | 15 | 16 | 46 | evaluación final externa (fotos propias) |

---

## 2. Reglas que no se deben romper

1. **No volver a dividir los datos.** Cargar siempre desde `splits.csv` filtrando por `split`.
   Si cada quien divide por su cuenta, los modelos no son comparables y puede haber fuga de información.
2. **`test` y `test_propias` se tocan solo al final**, una vez por modelo ya elegido.
   Hiperparámetros y decisiones se toman con `val` o con validación cruzada dentro de `train`.
3. **Validación cruzada dentro de `train`: usar `StratifiedGroupKFold` con la columna `grupo`.**
   Un `KFold`/`StratifiedKFold` normal vuelve a separar tomas casi iguales entre folds y el puntaje
   de CV sale inflado (la misma fuga que se corrigió en los splits).
4. **Aumento de datos, normalización y pesos de clase: solo se calculan con `train`.**
   Aumentar antes de dividir mete copias del mismo ejemplo en train y test.
5. **No usar accuracy como única métrica.** "regular" es ~3,9 % de los datos: un modelo que nunca
   la predice saca ~96 % de accuracy y no sirve. Reportar **macro-F1, precisión/recall por clase y
   matriz de confusión**.
6. **No ejecutar `prepare_dataset.py` con `--clean`** una vez hecho el reetiquetado de "regular":
   borra `processed` y se pierden esas imágenes. Para fotos nuevas usar `--append`.

---

## 3. Qué falta según el enunciado

### 3.1 Modelado (fase 4 CRISP-DM)
- **Línea base**: clasificador trivial (siempre la clase mayoritaria) y/o regresión logística simple.
  Sin línea base no se puede decir que el modelo "aprendió algo".
- **Al menos 2 modelos de ML tradicional**, p. ej. SVM y Random Forest/XGBoost, sobre características
  extraídas a mano (histogramas HSV, fracción de píxeles oscuros/manchas, textura). Ajustar
  hiperparámetros con `GridSearchCV(cv=StratifiedGroupKFold(...))` pasando `groups=grupo`, y registrar la rejilla probada.
- **Al menos 1 modelo de deep learning**: CNN pequeña (2–3 capas conv + pooling + densas),
  de preferencia **entrenada desde cero**. Si usan transfer learning, congelar capas y añadir capas propias.
- Opcional: un MLP sobre las mismas características manuales.
- **Desbalance de "regular"**: probar pesos de clase (`class_weight`) y/o aumento de datos solo en esa clase.
  Comparar con y sin, y documentar el efecto.
- **Tamaño**: el enunciado pide tamaño o diámetro normalizado. Opciones: clasificar `tamano`
  (pequeño/mediano/grande) o reportar el diámetro normalizado. **Decisión tomada:** el tamaño se *mide* con la
  máscara (`d_eq`, diámetro equivalente normalizado) y `tamano` son sus terciles por fruta; no hace falta un
  modelo de tamaño. **No usar `diam_ratio` ni `area_ratio`** (ver sección 4).
- **Entrenar sobre la fruta segmentada** (fondo pintado de gris con `aplicar_mascara`, o características
  calculadas solo dentro de la máscara). Ver `notebooks/03_data_preparation.ipynb`.
- Fijar semillas (`random_state`, `torch.manual_seed`/`tf.random.set_seed`) y guardar los modelos entrenados.

### 3.2 Evaluación (fase 5)
- Métricas en `test` y en `test_propias`, **por separado**. La diferencia entre ambas mide el
  cambio de dominio (Kaggle → fotos reales).
- Análisis de errores: mirar las imágenes mal clasificadas (¿se confunde regular con good o con bad?).
- Curvas de entrenamiento/validación para detectar sobreajuste.
- Comparar contra la línea base y, si se puede, contra resultados de la literatura.

### 3.3 Despliegue (fase 6)
- Interfaz sencilla (Streamlit es lo más rápido): cargar imagen o capturar con cámara y mostrar
  calidad + tamaño.
- **Debe aplicar exactamente el mismo preprocesamiento** que en `prepare_dataset.py`
  (corregir rotación EXIF, cuadrar con relleno del color de fondo, redimensionar a 224×224).
  Si el preprocesamiento difiere, el modelo fallará aunque en el notebook funcione.

### 3.4 Entregables
- Informe de máximo 7 páginas (estructura del enunciado, referencias IEEE, gráficos vectoriales).
- Diagrama de flujo personalizado de CRISP-DM.
- Análisis de aspectos éticos y de impacto.
- Video de máximo 10 minutos.
- Repositorio en GitHub organizado por fases, código documentado, datos de terceros referenciados
  (el dataset de Kaggle debe citarse explícitamente).

---

## 4. Limpieza de datos y limitaciones conocidas (deben decirse en el informe)

- **Duplicados en Kaggle.** Se eliminó cerca del 20 % por ser casi idénticas (dHash ≤ 10 de 256 bits,
  incluidas versiones espejadas) y con la misma etiqueta (`find_duplicates.py`).
- **Conflictos de etiqueta.** 31 parejas casi idénticas tenían etiquetas distintas: en todas, una era
  "regular" (reetiquetada a mano) y la otra su gemela good/bad. Se conservó la "regular" y se eliminó la
  gemela (`resolve_label_conflicts.py`, registro en `conflicts_removed.csv`). La foto propia de naranja
  duplicada quedó solo como "regular" (color irregular con parches verdes).
- **Split por grupos (fuga corregida).** Kaggle contiene ráfagas: varias fotos de la misma fruta tomadas
  con 1–2 s de diferencia y leves cambios de color o encuadre. Con el split por imagen había ~920 parejas de
  "la misma toma" repartidas entre train/val/test, lo que infla el test. Ahora `make_splits.py` agrupa
  (union-find) las fotos casi idénticas (dHash ≤ 20) y las consecutivas de una misma ráfaga (≤ 2 s según la
  hora del nombre original, o ≤ 5 en el contador de cámara, con dHash ≤ 60), y asigna cada grupo entero a un
  solo split, manteniendo ~70/15/15 por fruta y calidad. Resultado: 0 parejas con dHash ≤ 20 entre splits
  (antes 920). Quedan 57 parejas con dHash 21–30 entre splits; es fuga residual pequeña.
- **"regular" es una clase minoritaria y construida en parte a mano.** Kaggle solo trae good/bad.
  ~210 imágenes se reetiquetaron como regular tras revisión visual de candidatas; el resto son
  fotos propias. Además, varias de esas imágenes son de la misma ráfaga: hay solo **78 (apple),
  25 (banana) y 49 (orange) grupos independientes** con alguna "regular". Su test es pequeño
  (9–13 imágenes por fruta), así que las métricas por fruta de esa clase tienen mucha incertidumbre.
- **El tamaño no es un tamaño real.** `area_ratio` (umbral simple) resultó poco fiable: en las fotos propias
  vale ~0,6 cuando la fruta ocupa ~10 % de la imagen. Ahora se usa la máscara GrabCut: `d_eq` =
  2·sqrt(`mask_area`/π). Sigue siendo relativo a la foto: depende de la distancia de captura. Por eso 38 de
  las 46 fotos propias quedan como "pequeño" (se tomaron más lejos que las de Kaggle). En el despliegue hay
  que fijar la distancia de la cámara y calibrar los cortes.
- **Segmentación.** GrabCut con priors (`src/data/segmentacion.py`) elimina el atajo del fondo (ver
  notebook 03). En Kaggle la mano puede quedar parcialmente dentro de la máscara.
- **El fondo es un atajo en Kaggle** (ver `notebooks/02_data_understanding_eda.ipynb`, sección 4.1).
  Un Random Forest que solo ve el borde de la imagen logra F1 ≈ 0,89 en good y bad: muchas fotos de una
  misma clase comparten superficie. Por eso las características manuales deben calcularse **solo sobre la
  fruta**, la CNN necesita aumento de color y recorte (y conviene probar enmascarar el fondo), y
  `test_propias` es la prueba real de generalización. Además, en Kaggle la fruta suele estar en la mano: la
  segmentación simple por color del borde incluye la mano.
- **Solo 46 fotos propias.** Las métricas sobre `test_propias` son indicativas, no concluyentes.
- **Cambio de dominio.** Las imágenes de Kaggle y las fotos de celular/cámara web difieren en luz,
  fondo y encuadre; es esperable que el rendimiento baje en `test_propias`.

---

## 5. Cómo reproducir o ampliar los datos

Todos los comandos se ejecutan desde la raíz del repositorio.
Requiere `pip install -r requirements.txt` (incluye `pillow-heif` para las fotos `.HEIC`).

```powershell
# Primera vez (ya hecho)
python data\prepare_dataset.py --src data\fotos_originales --kaggle data\kaggle_raw --clean
python data\select_regular_candidates.py       # revisar data\review a mano
python data\select_regular_candidates.py --apply
python data\find_duplicates.py                 # revisar; luego con --mover
python data\find_duplicates.py --mover
python data\resolve_label_conflicts.py
python -m src.data.generar_mascaras            # ~20 min
python data\make_splits.py

# Agregar fotos nuevas más adelante
python data\prepare_dataset.py --src data\fotos_originales --append
python data\find_duplicates.py                 # solo reporta; revisar duplicates.csv
python -m src.data.generar_mascaras            # solo calcula las máscaras que faltan
python data\make_splits.py
```
Los splits de Kaggle no cambian si no cambian sus imágenes (semilla fija); solo crece `test_propias`.

---

## 6. Orden de trabajo sugerido

1. Cargar `splits.csv` y verificar conteos (reproducir la tabla de la sección 1).
2. EDA para el informe: distribución de clases por fruta, ejemplos por clase, resolución y brillo,
   distribución de `area_ratio`, tamaño de los grupos.
3. Línea base.
4. Características manuales + 2 modelos de ML con validación cruzada por grupos.
5. CNN pequeña desde cero.
6. Comparar todo en `val`, elegir, y evaluar una sola vez en `test` y `test_propias`.
7. Análisis de errores y de sobreajuste.
8. Interfaz Streamlit con el mismo preprocesamiento.
9. Informe, diagrama, video, repositorio.

## 7. Lista de verificación final

- [ ] Todo el código carga datos desde `splits.csv`
- [ ] Validación cruzada con `StratifiedGroupKFold` usando la columna `grupo`
- [ ] Métricas por clase (no solo accuracy) en `test` y `test_propias`
- [ ] Línea base y comparación con los modelos
- [ ] Hiperparámetros documentados (rejilla probada y valor elegido)
- [ ] Preprocesamiento de la interfaz idéntico al del entrenamiento
- [ ] Datos de terceros (Kaggle) y librerías citados
- [ ] Ética: sesgos de los datos, impacto en productores, errores y su costo
- [ ] Gráficos en formato vectorial, informe ≤ 7 páginas, video ≤ 10 min
