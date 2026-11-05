# Datos

| Carpeta | Contenido | ¿Versionado? |
|---|---|---|
| `raw/` | Dataset original *Fruit Quality Classification* (Kaggle) | No |
| `own_collection/` | 30–50+ fotos tomadas por el grupo (plaza, supermercado, casa) | Sí |
| `annotations/` | `labels.csv` con etiquetas de calidad y tamaño | Sí |
| `interim/` | Recortes individuales de `mixed_quality`, máscaras de segmentación | No |
| `processed/` | Imágenes redimensionadas / características listas para entrenar | No |

## Descarga del dataset

```bash
kaggle datasets download -d ryandpark/fruit-quality-classification -p data/raw --unzip
```

Requiere `kaggle.json` en `~/.kaggle/` (no subirlo al repositorio).

## Formato de anotaciones (`annotations/labels.csv`)

| columna | descripción |
|---|---|
| `filename` | ruta relativa a `data/` |
| `fruit` | tipo de fruta/verdura |
| `quality` | clase de calidad (definida en `docs/crisp_dm.md`) |
| `size` | pequeño / mediano / grande |
| `source` | `kaggle`, `kaggle_mixed`, `propia`, `otro_grupo` |
| `annotator` | quién etiquetó |

## Protocolo de captura de fotos propias

- Una sola fruta por foto, fondo simple y uniforme.
- Variar madurez, tamaño y defectos (golpes, manchas, podredumbre).
- Incluir un objeto de referencia o distancia fija para estimar tamaño.
- No capturar personas ni datos identificables (ver `docs/etica.md`).
