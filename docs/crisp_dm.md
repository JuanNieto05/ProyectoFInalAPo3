# Metodología CRISP-DM aplicada al proyecto

> Diagrama de flujo personalizado: `docs/figures/crisp_dm_flujo.svg` (pendiente).

## 1. Comprensión del negocio
- ¿Por qué es importante clasificar la calidad? Impacto económico y social.
- Definición de clases de calidad y criterios de tamaño.
- Métricas de éxito (p. ej. F1-macro, accuracy, matriz de confusión).

## 2. Comprensión de los datos
- Distribución de clases, desbalanceo, calidad y variabilidad de imágenes.

## 3. Preparación de los datos
- Segmentación de `mixed_quality`, redimensionado, normalización, aumento de datos, manejo de desbalanceo.

## 4. Modelado
- ML clásico (≥ 2 modelos) con validación cruzada / GridSearch.
- CNN simple desde cero.

## 5. Evaluación
- Métricas en test, análisis de errores, comparación con línea base.

## 6. Despliegue
- Interfaz gráfica con carga de imagen y cámara en tiempo real.
