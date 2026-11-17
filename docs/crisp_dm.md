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

Usaremos las frutas manzana, banano, guayaba, limón y naranja. Decidimos escoger frutas comunes y que tambien estén en Kaggle.

Se usarán 15 fotos de cada una como material propio junto con los registros del dataset de Kaggle.

## 3. Preparación de los datos
- Segmentación de `mixed_quality`, redimensionado, normalización, aumento de datos, manejo de desbalanceo.

Se puede encontrar el segment.py, donde se leen las imagenes, y se pasan al formato segmentado.

## 4. Modelado
- ML clásico (≥ 2 modelos) con validación cruzada / GridSearch.
- CNN simple desde cero.

## 5. Evaluación
- Métricas en test, análisis de errores, comparación con línea base.

## 6. Despliegue
- Interfaz gráfica con carga de imagen y cámara en tiempo real.
