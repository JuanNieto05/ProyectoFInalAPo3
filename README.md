# Clasificación automática de calidad de frutas y verduras con visión por computadora

Proyecto final del curso **Algoritmos y Programación III**, Ingeniería de Sistemas, Universidad Icesi (Cali, Colombia), semestre 2026-2.

#### -- Estado del proyecto: Activo

## Integrantes

**Líder del equipo: [Nombre completo](https://github.com/JuanNieto05)**
**Profesor: [Nombre completo](https://github.com/miltonsarria)**

| Nombre | GitHub | Correo |
|--------|--------|--------|
| [Nombre completo](https://github.com/usuario) | @usuario | correo@u.icesi.edu.co |
| [Nombre completo](https://github.com/usuario) | @usuario | correo@u.icesi.edu.co |
| [Nombre completo](https://github.com/usuario) | @usuario | correo@u.icesi.edu.co |

## Objetivo del proyecto

La clasificación manual de productos frescos según tamaño, madurez o defectos es lenta, subjetiva y propensa a errores, lo que genera pérdidas económicas, desperdicio de alimentos y falta de estandarización. Este proyecto desarrolla un sistema que, a partir de una fotografía de una fruta o verdura individual sobre fondo uniforme, **asigna una clase de calidad (≥ 3 categorías)** y **estima su tamaño relativo** (pequeño / mediano / grande o diámetro en píxeles normalizados), desplegado en una interfaz gráfica que funciona con imagen cargada o cámara en tiempo real.

### Métodos usados
* Análisis exploratorio de datos (EDA)
* Procesamiento y segmentación de imágenes
* Machine Learning clásico (≥ 2 modelos, con validación cruzada / búsqueda en rejilla)
* Deep Learning (CNN simple entrenada desde cero)
* Visualización de datos
* Metodología CRISP-DM

### Tecnologías
* Python, Jupyter
* NumPy, Pandas, OpenCV, scikit-image
* scikit-learn, XGBoost
* PyTorch / TensorFlow (por definir)
* Matplotlib, Seaborn
* Streamlit (despliegue)

## Metodología: mapa CRISP-DM → repositorio

| Fase CRISP-DM | Documentación | Notebook | Código |
|---|---|---|---|
| 1. Comprensión del negocio | [docs/crisp_dm.md](docs/crisp_dm.md), [docs/etica.md](docs/etica.md), [docs/impactos.md](docs/impactos.md) | `notebooks/01_business_understanding.ipynb` | — |
| 2. Comprensión de los datos | [data/README.md](data/README.md) | `notebooks/02_data_understanding_eda.ipynb` | `src/data/` |
| 3. Preparación de los datos | [docs/crisp_dm.md](docs/crisp_dm.md) | `notebooks/03_data_preparation.ipynb` | `src/data/`, `src/features/` |
| 4. Modelado | [docs/formulacion_matematica.md](docs/formulacion_matematica.md), [docs/arquitectura.md](docs/arquitectura.md) | `notebooks/04_modeling_ml_clasico.ipynb`, `notebooks/04_modeling_cnn.ipynb` | `src/models/`, `src/training/` |
| 5. Evaluación | [docs/formulacion_matematica.md](docs/formulacion_matematica.md) | `notebooks/05_evaluation.ipynb` | `src/evaluation/`, `experiments/results/` |
| 6. Despliegue | [docs/instalacion.md](docs/instalacion.md) | — | `app/` |

El diagrama de flujo personalizado de CRISP-DM está en [docs/crisp_dm.md](docs/crisp_dm.md).

## Estructura del repositorio

```
ProyectoFInalAPo3/
├── data/                 # Datos (ver data/README.md)
│   ├── raw/              # Dataset Kaggle original (no versionado)
│   ├── own_collection/   # Fotos recolectadas por el grupo
│   ├── annotations/      # Etiquetas (calidad + tamaño) en CSV
│   ├── interim/          # Datos intermedios (recortes, segmentación)
│   └── processed/        # Datos listos para entrenar (no versionado)
├── notebooks/            # Notebooks numerados por fase CRISP-DM
├── src/                  # Código fuente reutilizable
│   ├── data/             # Carga, segmentación y preprocesamiento
│   ├── features/         # Extracción de características (color, textura, forma)
│   ├── models/           # Definición de modelos (ML clásico y CNN)
│   ├── training/         # Scripts de entrenamiento y ajuste de hiperparámetros
│   ├── evaluation/       # Métricas, matrices de confusión, análisis de errores
│   ├── utils/            # Funciones auxiliares
│   └── config.py         # Rutas y constantes globales
├── app/                  # Interfaz gráfica (despliegue)
├── experiments/          # Resultados de experimentos
│   ├── logs/
│   ├── checkpoints/      # Modelos entrenados (no versionado)
│   └── results/          # Métricas y gráficas
├── docs/                 # Documentación: CRISP-DM, ética, impactos, matemáticas
├── reports/              # Informe final (≤ 7 págs., IEEE) y video
├── tests/                # Pruebas
├── requirements.txt
└── README.md
```

## Cómo empezar

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/JuanNieto05/ProyectoFInalAPo3.git
   ```
2. Crear el entorno e instalar dependencias (ver [docs/instalacion.md](docs/instalacion.md)):
   ```bash
   pip install -r requirements.txt
   ```
3. Descargar el dataset de Kaggle en `data/raw/` (ver [data/README.md](data/README.md)).
4. Ejecutar los notebooks en orden (`01_` → `05_`).
5. Lanzar la aplicación:
   ```bash
   streamlit run app/app.py
   ```

## Entregables

* Informe final: [reports/informe/](reports/informe/)
* Video (≤ 10 min): [reports/video/](reports/video/)
* Notebooks destacados: [notebooks/](notebooks/)

## Referencias y fuentes de terceros

* R. Park, *Fruit Quality Classification* [Dataset], Kaggle. Disponible: https://www.kaggle.com/datasets/ryandpark/fruit-quality-classification
* Plantilla de repositorio basada en [miltonsarria/ETLInfrati](https://github.com/miltonsarria/ETLInfrati).

Todo código o dato de terceros usado en el proyecto debe citarse aquí y en el informe (formato IEEE).
