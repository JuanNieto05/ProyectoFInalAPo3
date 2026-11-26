# Impactos de la solución (PI2)

En mercados y agroindustrias pequeñas, la fruta se clasifica a mano. Es un trabajo lento, y dos personas
pueden clasificar la misma fruta de forma distinta. Eso causa pérdidas de dinero y desperdicio de comida.
La FAO calcula que cerca del **14 % de los alimentos se pierde** entre la cosecha y la tienda [FAO 2019]. En
Colombia, el DNP estimó que se pierden o desperdician **9,76 millones de toneladas de alimentos al año**,
alrededor de un tercio de lo disponible, y las frutas y verduras son lo que más se pierde [DNP 2016].

Hicimos este análisis dos veces: una **al comienzo**, cuando planteamos la idea, y otra **al final**, con los
resultados en la mano. Las dos versiones son distintas, porque varias cosas que esperábamos no se
cumplieron y aparecieron riesgos que no habíamos pensado.

## 1. Lo que esperábamos al comienzo

| Dimensión | Lo bueno que esperábamos | Lo que nos preocupaba | Cómo pensábamos evitarlo |
|---|---|---|---|
| **Social** | Un criterio de calidad parejo, menos trabajo repetitivo y fruta más uniforme para el cliente | Que la cámara se sintiera como vigilancia, fotografiar personas, quitarle el trabajo a quien clasifica | Apuntar la cámara solo a la fruta, no fotografiar personas y diseñarlo como una ayuda |
| **Económica** | Clasificar más rápido y más barato, y vender mejor la fruta bien clasificada | Que los errores hicieran botar fruta buena o vender fruta mala, y el costo de los equipos | Medir cada tipo de error y usar equipos comunes (un celular o un portátil) |
| **Ambiental** | Menos comida desperdiciada: la fruta *regular* se vende con descuento en vez de botarse | La energía que gasta procesar imágenes y entrenar modelos | Usar modelos pequeños que funcionen sin tarjeta gráfica |
| **Global** | Que sirviera en cualquier mercado con una cámara | Que los datos fueran de otro país y que cada país tenga su propia idea de "buena calidad" | Agregar fotos nuestras y escribir claramente qué entendemos por cada clase |

## 2. Lo que encontramos al final

### Social

**Lo bueno.**
- El sistema **muestra lo que está mirando**: dibuja el contorno de la fruta que analizó y avisa cuando no
  está seguro. No es una caja negra.
- Con fotos parecidas a las de Kaggle, distingue bien las frutas buenas de las malas, con un acierto de
  alrededor de 0,9.

**Los riesgos.**
- **Privacidad.** 13 de nuestras fotos tenían guardada la ubicación GPS. No lo habíamos previsto; ya la
  borramos (ver `docs/etica.md`).
- **Confianza mal puesta.** Con nuestras fotos, el modelo se muestra igual de seguro cuando se equivoca que
  cuando acierta. Alguien podría creerle demasiado.
- **Empleo.** Si se usara para reemplazar a quien clasifica, ese trabajo desaparecería. Y con nuestros
  resultados, el reemplazo además lo haría peor.

**Qué hacemos al respecto.** La app avisa cuando el resultado es dudoso y recomienda revisar a mano. La
presentamos como una **ayuda**: la persona decide y el sistema le acelera los casos fáciles.

### Económica

**Lo bueno.**
- **Es muy barato de usar.** Funciona en un portátil normal, sin tarjeta gráfica, y tarda unos 0,15
  segundos por fruta.
- El modelo pesa menos de 1 MB, y todo el software que usamos es gratuito.

**Los riesgos.**
- **Los errores tienen costos distintos.** En las pruebas, el modelo principal (SVM):
  - deja pasar como buena el **4 %** de la fruta mala, que le llega dañada al cliente;
  - marca como mala el **14 %** de la fruta buena, que se bota o se vende barata.

  La red neuronal hace lo contrario: deja pasar menos fruta mala (1 %), pero rechaza más fruta buena (25 %).
- **La fruta *regular* es la que más dinero podría salvar**, porque se puede vender con descuento en vez de
  botarla. Pero es justo la que el sistema peor reconoce: con nuestras fotos, acertó 1 de 15.

**Qué hacemos al respecto.** Elegir el modelo según qué error le duela más a quien lo use. Si lo peor es
venderle fruta mala a un cliente, conviene la red neuronal; si lo peor es botar fruta buena, el SVM. Para
mejorar con *regular* hacen falta más fotos y un criterio en el que varias personas estén de acuerdo.

### Ambiental

**Lo bueno: gasta muy poca energía.**
- Entrenar todas nuestras redes neuronales tomó unas 6 horas en un portátil, unos 0,2 a 0,3 kWh: más o
  menos lo que gasta un televisor encendido una tarde.
- Clasificar 10.000 frutas al día gastaría unos 13 Wh, menos que un bombillo LED prendido dos horas.
- No necesita servidores ni equipos especiales.

**El riesgo: no podemos asegurar que reduzca el desperdicio.** Ese era el beneficio ambiental más
importante, pero depende de reconocer bien la fruta *regular* y de no botar fruta buena, que es justo donde
el sistema falla más. Un sistema que rechaza el 14 % de la fruta buena podría **aumentar** el desperdicio en
lugar de reducirlo.

**Qué hacemos al respecto.** No usarlo para botar fruta de forma automática. Antes de usarlo de verdad,
hacer una prueba en un mercado y comparar cuánta fruta se desperdicia con y sin el sistema.

### Global

**Lo bueno.** Todo está publicado (código, datos y forma de tomar las fotos), así que otro grupo podría
repetir el trabajo con su propia fruta. Además, lo que aprendimos sobre el fondo de las fotos y sobre las
fotos repetidas sirve para cualquier proyecto con imágenes.

**Los riesgos.**
- **No funciona igual fuera de su contexto.** Con fotos de Kaggle acierta 0,72; con las nuestras, 0,47. La
  diferencia entre las fotos pesa más que la diferencia entre modelos. Un sistema entrenado en un país y
  usado en otro clasifica peor.
- **"Calidad" no significa lo mismo en todas partes.** En Kaggle, casi todas las naranjas verdes están
  marcadas como buenas. Para nosotros eran *regular*, aunque en Colombia muchas naranjas se venden verdes.
- **Cada país tiene sus propias normas de calidad** para frutas (en Colombia, las del ICONTEC), y este
  sistema no las sigue.

**Qué hacemos al respecto.**
- Reentrenar con fotos del lugar donde se vaya a usar.
- Ajustar las clases a la norma de cada país.
- Compartir fotos entre grupos para tener datos más variados, como propone el enunciado.

## 3. Qué cambió entre el comienzo y el final

1. **El beneficio ambiental pasó de "esperado" a "no demostrado".** Creíamos que clasificar mejor reduciría
   el desperdicio, pero el sistema falla justo en la fruta que se podría salvar.
2. **El riesgo de privacidad apareció en otro lugar.** Nos preocupaba la cámara en vivo, y el problema real
   estaba en los datos ocultos de las fotos.
3. **Lo global resultó ser lo más importante.** Lo que más bajó los resultados no fue el modelo sino el
   cambio de fotos: otra luz, otro fondo, otra cámara y otra idea de madurez.
4. **El gasto de energía fue mucho menor de lo que temíamos.** Elegir modelos pequeños fue también una
   buena decisión ambiental y económica.

## 4. Conclusión

Hoy el sistema sirve como **ayuda** para separar fruta buena de fruta mala cuando las fotos se parecen a
las de entrenamiento. **No sirve para decidir solo**, ni para usarse en otro lugar sin antes ajustarlo.

Antes de usarlo en un mercado real habría que:
- probarlo con fotos de ese mercado;
- ponerse de acuerdo con quienes clasifican hoy sobre qué es *regular*;
- medir si de verdad ahorra tiempo y reduce el desperdicio.

## Referencias de esta sección

- [FAO 2019] FAO, *The State of Food and Agriculture 2019. Moving forward on food loss and waste reduction*.
  Rome: FAO, 2019.
- [DNP 2016] Departamento Nacional de Planeación, *Pérdida y desperdicio de alimentos en Colombia*. Bogotá:
  DNP, 2016.
- Todos los resultados citados están en `notebooks/05_evaluation.ipynb` y en `experiments/results/`.
