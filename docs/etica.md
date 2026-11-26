# Análisis ético (PI1)

Nuestro sistema mira la foto de una fruta y dice si está buena, regular o mala, y si es pequeña, mediana o
grande. La idea es que **ayude** a quien clasifica fruta en un mercado, no que lo reemplace ni que decida
solo.

Al construirlo nos encontramos con varias decisiones que podían afectar a otras personas. Para revisarlas
usamos tres referencias:
- el **Código de Ética de la ACM** (2018);
- el **Código de Ética del IEEE** (2020);
- en Colombia, la **Ley 842 de 2003**, que es el código de ética de los ingenieros, y la **Ley 1581 de
  2012**, sobre protección de datos personales.

## 1. Las situaciones que encontramos

| # | Qué pasó en el proyecto | Por qué es un problema ético | Principio relacionado | Qué hicimos |
|---|---|---|---|---|
| 1 | Tomamos las fotos propias con celulares (iPhone) | Al revisarlas descubrimos que **13 de las 47 fotos guardaban la ubicación GPS** de donde se tomaron, por ejemplo la casa de un integrante. Esa información iba a quedar pública en GitHub. | Respetar la privacidad (ACM 1.6, IEEE I.1, Ley 1581) | Las imágenes que usan los modelos nunca tuvieron esos datos. En las 13 fotos originales los borramos todos (GPS, modelo del celular y fecha). La foto se ve igual y los resultados no cambian. |
| 2 | Fuimos a tomar fotos a mercados y a la casa | Se podía fotografiar sin querer a personas o datos que las identifiquen, sin su permiso | Privacidad y no causar daño (ACM 1.2 y 1.6) | Definimos reglas para tomar las fotos: una fruta, fondo liso, **sin personas**. Las fotos solo se usan para el curso. |
| 3 | Tuvimos que inventar la clase *regular*, porque Kaggle solo trae buenas y malas | Decidir qué es "regular" es subjetivo. Un criterio mal pensado puede hacer que se bote fruta sana o que se venda fruta dañada. | Ser honestos y rigurosos (ACM 1.3 y 2.5) | Escribimos el criterio con porcentajes de daño y una regla para los casos dudosos: si una fruta está entre dos clases, se pone la peor. También contamos dónde falló el criterio (las naranjas verdes). |
| 4 | Casi todas las fotos de entrenamiento son de Kaggle (5.659 de 5.705) | El modelo aprende cómo se ve la fruta de esas fotos. Con fruta de aquí, por ejemplo la de un pequeño productor en la plaza, funciona peor. | No discriminar (ACM 1.4) | Lo medimos: funciona bastante peor con nuestras fotos (0,47 frente a 0,72). Lo decimos en la app y en el informe, sin esconderlo. |
| 5 | En Kaggle, el fondo de la foto delata si la fruta es buena o mala | Un modelo podría "acertar" mirando el fondo y no la fruta. Una fruta mala sobre un fondo "de buenas" pasaría como buena. | Evaluar con honestidad los riesgos (ACM 2.5) | Lo detectamos en el análisis de datos, recortamos la fruta del fondo y luego comprobamos con nuestras fotos si el problema seguía. |
| 6 | Había fotos casi iguales repartidas entre los datos de entrenamiento y de prueba | Los resultados iban a salir mejores de lo que de verdad son, y eso engaña a quien quiera usar el sistema | Ser honestos con lo que el sistema puede hacer (ACM 1.3, IEEE I.5) | Agrupamos las fotos de una misma toma para que no quedaran repartidas: 920 parejas casi iguales bajaron a 0. Probamos con los datos de prueba **una sola vez**. |
| 7 | Los resultados con nuestras fotos no llegaron a la meta (0,47 frente a 0,70) | Era tentador mostrar solo los buenos resultados o ajustar el modelo hasta que salieran mejor | Integridad y veracidad (ACM 1.3, IEEE I.5, Ley 842) | Reportamos todo, con márgenes de error. No cambiamos el modelo después de ver los resultados finales. |
| 8 | El sistema se podría usar para rechazar fruta | Si dice "mala" a una fruta buena, se bota comida y se pierde dinero. Si dice "buena" a una mala, le llega fruta dañada al cliente. | No causar daño y cuidar el bienestar (ACM 1.2, IEEE I.1) | Medimos cada error por separado: el SVM deja pasar el 4 % de la fruta mala y rechaza el 14 % de la buena. La app dice que es una **ayuda, no la decisión final**. |
| 9 | La app muestra qué tan seguro está el modelo | Esa "seguridad" puede engañar: con nuestras fotos, el modelo está igual de seguro cuando se equivoca que cuando acierta | Comunicar las limitaciones con claridad (ACM 1.3 y 2.5) | La app avisa **"Resultado dudoso"** cuando el modelo duda, y muestra qué parte de la foto analizó para que la persona lo pueda revisar. |
| 10 | El tamaño se calcula en píxeles | El tamaño en la foto cambia según qué tan cerca esté la cámara. Si alguien lo usara para poner precios, podría pagar menos por una fruta grande fotografiada de lejos. | Ser honestos con las limitaciones (ACM 1.3) | La app pide tomar la foto **siempre a la misma distancia** y lo explica en pantalla. |
| 11 | Usamos datos y código de otras personas | Usar trabajo ajeno sin citarlo es plagio, y el curso lo considera fraude | Dar crédito y respetar licencias (ACM 1.5 y 2.3, IEEE I.5) | Citamos el dataset de Kaggle, las librerías y los métodos en el README y en el informe. Revisamos la licencia del dataset: es **CC0 (dominio público)**, así que se puede usar y modificar libremente; igual lo citamos. |
| 12 | Usamos herramientas de inteligencia artificial para programar y redactar | Presentar como propio algo hecho con ayuda de IA no es honesto | Honestidad y crédito (ACM 1.3 y 1.5) | Lo declaramos en el informe. Todo lo que produjo la IA lo revisó el equipo. |

## 2. Lo que aprendimos

**La honestidad fue el tema central.** Varias de nuestras decisiones más trabajosas hicieron que los
resultados se vieran *peores*, pero más reales: separar las fotos por toma, recortar el fondo y probar una
sola vez. Preferimos mostrar un 0,47 verdadero que un 0,90 inflado.

**Los errores no cuestan lo mismo.** Botar fruta buena le cuesta al vendedor; vender fruta mala le cuesta
al cliente. Por eso no basta con un solo número de "qué tan bien funciona". Hay que mirar cada tipo de
error.

**Un sistema puede ser injusto sin quererlo.** Si se entrena con fruta de un solo lugar, funciona peor con
la de otro lugar. Eso puede perjudicar justo a quien vende fruta local.

**La privacidad estaba donde no la esperábamos.** Nos preocupaba la cámara en vivo, pero el riesgo real
eran los datos escondidos dentro de las fotos que subimos.

## 3. Lo que revisamos antes de entregar

1. **Licencia del dataset de Kaggle:** es CC0 (dominio público). La citamos en las referencias.
2. **Uso de herramientas de IA:** lo declaramos en el informe y en el README.
3. **Ubicación GPS:** la borramos de las 13 fotos originales que la tenían. Ojo: las versiones viejas
de esas fotos siguen en el historial de git. Si el repositorio va a ser público, el equipo puede decidir si
también se limpia el historial.

## 4. Recomendaciones para quien quiera usar el sistema

- Úselo como una **segunda opinión**. Las fotos marcadas como dudosas debe revisarlas una persona.
- Antes de confiar en él, **pruébelo con fotos de su propio lugar**: su luz, su fondo y su cámara. Nuestros
  resultados muestran que, sin esto, se equivoca mucho más.
- No use el tamaño para fijar precios, a menos que tome las fotos siempre a la misma distancia.
- Si va a clasificar fruta *regular*, primero póngase de acuerdo con su equipo sobre qué significa
  "regular". Si dos personas no lo clasifican igual, el sistema tampoco podrá hacerlo bien.
