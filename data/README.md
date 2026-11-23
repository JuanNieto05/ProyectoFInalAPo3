# Datos

Tres frutas (`apple`, `banana`, `orange`) × tres calidades (`good`, `regular`, `bad`): **5.705 imágenes**
(5.659 de Kaggle + 46 propias). Detalle del proceso y limitaciones: [docs/GUIA_SIGUIENTE_ETAPA.md](../docs/GUIA_SIGUIENTE_ETAPA.md).

| Carpeta / archivo | Contenido | ¿Versionado? |
|---|---|---|
| `fotos_originales/{Apple,Banana,Orange}/{good,regular,bad}/` | Fotos propias originales (respaldo) | Sí |
| `processed/{good,regular,bad}/` | Imágenes finales 224×224 RGB (Kaggle + propias). La carpeta es la clase. | Sí |
| `manifest.csv` | Una fila por imagen: fruta, calidad, fuente, original, `area_ratio`, `diam_ratio`, `relabeled_from` | Sí |
| `splits.csv` | **Fuente única de verdad**: split, `grupo` (misma toma) y `tamano` | Sí |
| `duplicates.csv` | Parejas de duplicados detectadas por dHash | Sí |
| `conflicts_removed.csv` | Imágenes eliminadas por conflicto de etiqueta | Sí |
| `kaggle_raw/` | Dataset original de Kaggle descomprimido | No |
| `review/`, `duplicates_removed/`, `conflicts_removed/`, `duplicates_preview/` | Carpetas de trabajo / imágenes retiradas | No |

## Descarga del dataset de Kaggle

```bash
kaggle datasets download -d ryandpark/fruit-quality-classification -p data/kaggle_raw --unzip
```

Requiere `kaggle.json` en `~/.kaggle/` (no subirlo al repositorio).

## Scripts (ejecutar desde la raíz del repo, en este orden)

| Script | Qué hace |
|---|---|
| `prepare_dataset.py` | Corrige rotación EXIF, cuadra con el color de fondo, redimensiona a 224×224, calcula `area_ratio`/`diam_ratio` y escribe `manifest.csv` |
| `select_regular_candidates.py` | Propone imágenes de Kaggle candidatas a "regular" para revisión manual y aplica el reetiquetado (`--apply`) |
| `find_duplicates.py` | Detecta duplicados por dHash (≤ 10 bits, incluye espejo); `--mover` elimina los de igual etiqueta |
| `resolve_label_conflicts.py` | Resuelve duplicados con etiqueta distinta (conserva la "regular") |
| `make_splits.py` | Split 70/15/15 estratificado **por grupos** (misma toma) + `test_propias` + `tamano` |

## Columnas de `splits.csv`

| columna | descripción |
|---|---|
| `filename`, `path` | nombre y ruta relativa a `data/processed/` |
| `fruit` | `apple`, `banana`, `orange` |
| `quality` | `good`, `regular`, `bad` |
| `source` | `kaggle` o `own` |
| `split` | `train`, `val`, `test`, `test_propias` |
| `grupo` | id del grupo de fotos de la misma toma (usar en `StratifiedGroupKFold`) |
| `tamano` | `pequeno` / `mediano` / `grande` (terciles de `area_ratio` por fruta, cortes con train) |
| `area_ratio` | fracción de la imagen ocupada por la fruta (proxy de tamaño) |
| `diam_ratio` | lado mayor del bounding box / lado de la imagen (≈1,0 casi siempre: no usar) |

## Protocolo de captura de fotos propias

- Una sola fruta por foto, fondo simple y uniforme.
- Variar madurez, tamaño y defectos (golpes, manchas, podredumbre).
- Distancia cámara–fruta aproximadamente fija (o un objeto de referencia) para que el tamaño sea comparable.
- No capturar personas ni datos identificables (ver `docs/etica.md`).

## Criterios de calidad (3 clases)

| Clase | Criterio |
|---|---|
| **good** (buena) | Piel íntegra, color uniforme y propio de fruta madura, sin golpes visibles. Defectos menores en menos de ~5 % de la superficie visible. |
| **regular** | Defectos visibles moderados (manchas, golpes leves, color irregular, muy verde o pasada) en ~5 a 25 % de la superficie. Aún vendible con descuento. |
| **bad** (mala) | Defectos en más de ~25 %, podredumbre, moho, piel rota o deformación. No apta para la venta. |

**Regla de duda:** si una foto cae entre dos clases, se elige la de menor calidad.

## Fuente de terceros

R. Park, *Fruit Quality Classification* [Dataset], Kaggle. https://www.kaggle.com/datasets/ryandpark/fruit-quality-classification
