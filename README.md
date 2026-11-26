# Clasificación de calidad y tamaño de frutas con visión por computadora

Proyecto final del curso **Algoritmos y Programación III**, Ingeniería de Sistemas, Universidad Icesi (Cali, Colombia), semestre 2026-2.

**Estado del proyecto:** terminado.

## Integrantes

* Juan Felipe Nieto
* Laura Sofía Buitrago Sánchez
* Jose David Loaiza

## De qué se trata

En los mercados la fruta se clasifica a mano. Es lento, cada persona tiene su propio criterio y se desperdicia comida.
Construimos un sistema que recibe la foto de una fruta (manzana, banano o naranja) y dice dos cosas:

* **su calidad:** buena, regular o mala;
* **su tamaño:** pequeña, mediana o grande, comparada con otras frutas del mismo tipo.

Todo se usa desde una aplicación web que acepta una foto de la cámara o un archivo.

## Resultados principales

Usamos **macro-F1** como métrica principal: es el promedio del F1 de las tres clases, así que la clase "regular",
que es la más escasa, pesa lo mismo que las otras dos.

| Modelo | Validación | Test (Kaggle) | Fotos propias (46) |
|---|---|---|---|
| **SVM** (elegido) | 0,727 | 0,723 | 0,470 |
| CNN desde cero | 0,648 | 0,647 | 0,382 |
| Línea base (azar o clase mayoritaria) | 0,22–0,31 | — | — |

Lo más importante que aprendimos:

* **El fondo delataba la clase.** En Kaggle, las frutas buenas y malas se fotografiaron sobre fondos distintos, y un
  modelo podía acertar mirando solo el borde de la imagen. Por eso separamos la fruta del fondo (GrabCut) antes de
  entrenar.
* **Con nuestras fotos el rendimiento baja mucho** (de 0,72 a 0,47). Las fotos propias son más oscuras y la fruta
  sale más pequeña. El sistema sirve como apoyo, no para reemplazar a una persona.
* **El tamaño depende de la distancia de la cámara.** Al compararlo con nuestra etiqueta manual coincide en el 61 %
  de las fotos (kappa 0,31). Para que funcione bien hay que fotografiar siempre a la misma distancia.

El análisis completo está en el informe y en `notebooks/05_evaluation.ipynb`.

## Métodos y tecnologías

* **Metodología:** CRISP-DM.
* **Datos:** dataset de Kaggle y 46 fotos tomadas por el equipo. División 70/15/15 **por grupos de toma**, para que
  fotos casi iguales de la misma fruta no queden a la vez en entrenamiento y en prueba.
* **Preparación:** segmentación con GrabCut y 80 características de color, textura y forma sobre la fruta.
* **Machine Learning clásico:** 5 modelos ajustados con GridSearch y validación cruzada por grupos (el mejor es el
  SVM).
* **Deep Learning:** una CNN pequeña entrenada desde cero en PyTorch.
* **Herramientas:** Python, NumPy, Pandas, OpenCV, scikit-image, scikit-learn, XGBoost, PyTorch, Matplotlib,
  Seaborn, Jupyter y Streamlit.

## CRISP-DM en el repositorio

| Fase | Documentación | Notebook | Código |
|---|---|---|---|
| 1. Comprensión del negocio | [crisp_dm.md](docs/crisp_dm.md), [etica.md](docs/etica.md), [impactos.md](docs/impactos.md) | — | — |
| 2. Comprensión de los datos | [data/README.md](data/README.md) | `02_data_understanding_eda.ipynb` | `src/data/` |
| 3. Preparación de los datos | [data/README.md](data/README.md), [crisp_dm.md](docs/crisp_dm.md) | `03_data_preparation.ipynb` | `data/*.py`, `src/data/`, `src/features/` |
| 4. Modelado | [formulacion_matematica.md](docs/formulacion_matematica.md), [arquitectura.md](docs/arquitectura.md) | `04_modeling_ml_clasico.ipynb`, `04_modeling_cnn.ipynb` | `src/models/`, `src/training/` |
| 5. Evaluación | [formulacion_matematica.md](docs/formulacion_matematica.md) | `05_evaluation.ipynb` | `src/evaluation/`, `experiments/results/` |
| 6. Despliegue | [instalacion.md](docs/instalacion.md) | — | `app/app.py` |

El diagrama personalizado del ciclo, con las veces que tuvimos que volver atrás y por qué, está en
[docs/crisp_dm.md](docs/crisp_dm.md).

## Estructura del repositorio

```
ProyectoFInalAPo3/
├── app/
│   └── app.py              # Aplicación Streamlit (cámara o archivo)
├── data/                   # Datos y pipeline de preparación (ver data/README.md)
│   ├── fotos_originales/   # Fotos tomadas por el equipo, tal como salieron de la cámara
│   ├── processed/          # Imágenes 224×224 por clase (good / regular / bad)
│   ├── masks/              # Máscaras de la fruta (GrabCut)
│   ├── splits.csv          # Fuente única de verdad: split, grupo de toma y tamaño
│   ├── tamano_manual.csv   # Etiqueta manual de tamaño de las fotos propias
│   └── *.py                # Scripts de preparación
├── notebooks/              # Un notebook por fase de CRISP-DM (02 a 05)
├── src/
│   ├── data/               # Carga, segmentación y tensores para la CNN
│   ├── features/           # Extracción de características
│   ├── models/             # Modelos clásicos y CNN
│   ├── training/           # Entrenamiento de la CNN
│   ├── evaluation/         # Métricas, predicción y comparación del tamaño manual
│   ├── utils/              # Diagrama de CRISP-DM
│   └── config.py           # Rutas y constantes
├── experiments/
│   ├── results/            # Métricas y figuras de los experimentos
│   └── checkpoints/        # Modelos entrenados (no se suben a Git)
├── docs/                   # CRISP-DM, ética, impactos, matemáticas, arquitectura, instalación y referencias
├── reports/
│   ├── informe/            # Informe final en formato IEEE (6 páginas)
│   └── video/              # Video de presentación
└── requirements.txt
```

## Cómo ejecutarlo

La guía completa está en [docs/instalacion.md](docs/instalacion.md). En resumen:

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/JuanNieto05/ProyectoFInalAPo3.git
   cd ProyectoFInalAPo3
   ```
2. Crear el entorno e instalar las dependencias (PyTorch para CPU primero):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt
   ```
3. Generar los modelos. No están en Git porque pesan y dependen de la versión de scikit-learn:
   ```bash
   python -m nbconvert --to notebook --execute --inplace notebooks/04_modeling_ml_clasico.ipynb
   ```
   Esto crea `experiments/checkpoints/ml_mejor.joblib`, que es el que usa la app.
4. Abrir la aplicación:
   ```bash
   streamlit run app/app.py
   ```

Los datos procesados ya están en el repositorio. Kaggle solo hace falta para regenerarlos desde cero (ver
[data/README.md](data/README.md)).

## Entregables

* **Informe** (IEEE, 6 páginas): [reports/informe/informe.docx](reports/informe/informe.docx)
* **Video** (≤ 10 min): [reports/video/](reports/video/)
* **Notebooks:** [notebooks/](notebooks/)

## Referencias y fuentes de terceros

* R. Park, *Fruit Quality Classification* [Dataset], Kaggle. Disponible en:
  https://www.kaggle.com/datasets/ryandpark/fruit-quality-classification
* Plantilla de repositorio basada en [miltonsarria/ETLInfrati](https://github.com/miltonsarria/ETLInfrati).
* Lista completa en formato IEEE: [docs/referencias.md](docs/referencias.md).

