# Metodología CRISP-DM aplicada al proyecto

> Diagrama de flujo personalizado: `docs/figures/crisp_dm_flujo.svg` (pendiente).

## 1. Comprensión del negocio
- ¿Por qué es importante clasificar la calidad? Impacto económico y social.

Es importante clasificar la calidad porque la clasificación manual de frutas y verduras depende del error y ritmo humano. Además, este sector requiere recortar costos y tiempo de producción, pues es un sector vital para la economia y el dia a dia y bolsillo de los colombianos.

- Definición de clases de calidad y criterios de tamaño.

Decidimos usar 3 clases de calidad: Buena, regular, mala. 

Cada una basada en el aspecto de la fruta, su madurez, manchas y golpes. 

- Métricas de éxito (p. ej. F1-macro, accuracy, matriz de confusión).

Usaremos F1 macro, accuracy, precision y recall de la clase "mala" y kappa.

F1 macro ≥ 0,85 en Kaggle y ≥ 0,70 en fotos propias.

Especificamente, para la parte de la calidad de los datos se usará Kappa de Cohen, cobertura e índice de desbalance.

En el despliegue usaremos latencia por imagen, tamaño del modelo y calibración de la confianza.

## 2. Comprensión de los datos
- Distribución de clases, desbalanceo, calidad y variabilidad de imágenes.

Usamos tres frutas: manzana (`apple`), banano (`banana`) y naranja (`orange`). Escogimos frutas comunes en Colombia que también están en Kaggle.

Datos: 5.659 imágenes de Kaggle (tras limpieza) + 46 fotos propias (~5 por fruta y calidad). Kaggle solo trae good/bad; la clase `regular` se construyó reetiquetando a mano 210 imágenes candidatas, más las fotos propias. Total 5.705 imágenes; `regular` es ~3,9 % (desbalance fuerte).

## 3. Preparación de los datos
- Segmentación de `mixed_quality`, redimensionado, normalización, aumento de datos, manejo de desbalanceo.

Pipeline (scripts en `data/`, detalle en `data/README.md` y `docs/GUIA_SIGUIENTE_ETAPA.md`):
1. `prepare_dataset.py`: corrección EXIF, cuadrado con relleno del color de fondo, 224×224, estimación de `area_ratio`.
2. `select_regular_candidates.py`: candidatas a `regular` + revisión manual.
3. `find_duplicates.py`: elimina ~20 % de duplicados (dHash ≤ 10, con espejo).
4. `resolve_label_conflicts.py`: elimina 31 gemelas con etiqueta contradictoria.
5. `make_splits.py`: split 70/15/15 estratificado **por grupos de la misma toma** (dHash + ráfagas por hora de captura) para evitar fuga; fotos propias como `test_propias`.

## 4. Modelado
- ML clásico (≥ 2 modelos) con validación cruzada / GridSearch.
- CNN simple desde cero.

## 5. Evaluación
- Métricas en test, análisis de errores, comparación con línea base.

## 6. Despliegue
- Interfaz gráfica con carga de imagen y cámara en tiempo real.
