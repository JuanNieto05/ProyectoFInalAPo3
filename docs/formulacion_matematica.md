# Formulación matemática y solución (PI3)

Este documento explica el problema en términos matemáticos y por qué resolvimos cada parte como lo hicimos.
Lo organizamos en los cinco puntos que pide el curso. Cada fórmula viene con una explicación en palabras y
con el archivo donde está programada.

---

## 1. El problema

### 1.1 Qué entra y qué sale

**Entra** una foto de una fruta. Para el computador, una foto es una tabla de números: cada píxel tiene
tres valores (rojo, verde y azul) entre 0 y 255. Primero la dejamos cuadrada y de 224 × 224 píxeles, así
que la foto es

$$
I \in \{0, \dots, 255\}^{224 \times 224 \times 3}
$$

**Salen** dos cosas:
- la **calidad** $\hat y$: buena, regular o mala;
- el **tamaño** $\hat t$: pequeño, mediano o grande.

El usuario también dice qué fruta es (manzana, banano o naranja), porque una manzana grande y un banano
grande no miden lo mismo.

### 1.2 Cómo lo dividimos en pasos

Resolvimos el problema en tres pasos, uno detrás de otro:

$$
\underbrace{M = S(I)}_{\text{1. encontrar la fruta}} \qquad
\underbrace{\hat y = f\big(\phi(I, M)\big)}_{\text{2. decidir la calidad}} \qquad
\underbrace{\hat t = T\big(d_{eq}(M)\big)}_{\text{3. medir el tamaño}}
$$

1. **Encontrar la fruta.** $S$ separa la fruta del fondo y devuelve una **máscara** $M$: una imagen en blanco
   y negro donde 1 es fruta y 0 es fondo. (`src/data/segmentacion.py`)
2. **Decidir la calidad.** $\phi$ convierte la fruta en **80 números** que la describen (color, manchas,
   textura, forma), y el modelo $f$ decide la calidad a partir de esos números. En la red neuronal, en
   cambio, se le pasa directamente la foto de la fruta sobre un fondo gris. (`src/features/extraccion.py`)
3. **Medir el tamaño.** Con el área de la máscara calculamos un diámetro y lo comparamos con el de otras
   frutas del mismo tipo.

### 1.3 Qué queremos lograr

Si solo midiéramos cuántas frutas acierta el modelo, podríamos engañarnos. La fruta *regular* es apenas el
**3,8 %** de los datos de entrenamiento (146 de 3.884), así que un modelo que **nunca** dijera "regular"
igual acertaría el 96 % de las veces.

Por eso usamos el **macro-F1**, que le da el mismo peso a las tres clases:

$$
\text{macro-F1} = \frac{F1_{\text{buena}} + F1_{\text{regular}} + F1_{\text{mala}}}{3}
$$

El **F1** de una clase combina dos preguntas:
- **precisión**: de las frutas que el modelo llamó "regular", ¿cuántas lo eran?
- **recall**: de las frutas que de verdad eran "regular", ¿cuántas encontró?

$$
F1 = \frac{2 \cdot \text{precisión} \cdot \text{recall}}{\text{precisión} + \text{recall}}
$$

Nos pusimos como meta un macro-F1 de al menos **0,85 con fotos de Kaggle** y **0,70 con nuestras fotos**. Lo
que de verdad importa es cómo le va con fotos nuevas, como las que tomaría alguien en un mercado; por eso
guardamos nuestras 46 fotos solo para la prueba final.

### 1.4 Por qué es un problema difícil

1. **Hay muy pocas frutas *regular*,** y además la diferencia entre "regular" y "buena" es difícil de ver
   incluso para una persona.
2. **El fondo hace trampa.** En Kaggle, el fondo de la foto delata la calidad: un modelo que solo mira los
   bordes de la imagen, sin ver la fruta, acierta casi el 90 %.
3. **Había fotos repetidas.** Kaggle tiene ráfagas de fotos casi iguales. Si una queda para entrenar y su
   "gemela" para probar, el modelo parece mejor de lo que es. Encontramos 920 casos así.
4. **Nuestras fotos son distintas a las de Kaggle:** más oscuras (brillo 129 frente a 162), con la fruta más
   pequeña dentro de la foto y con otra idea de madurez.
5. **En una foto no hay centímetros, solo píxeles.** El tamaño de la fruta en la foto cambia según qué tan
   lejos esté la cámara.
6. **Hay dos objetivos a la vez** (calidad y tamaño) y todo tiene que funcionar rápido en un computador
   normal.

### 1.5 El proceso en pseudocódigo

```
PREPARAR Y ENTRENAR
  juntar fotos de Kaggle y propias, dejarlas de 224×224
  quitar fotos repetidas y con etiquetas contradictorias
  agrupar las fotos de una misma toma
  repartir por grupos: 70 % entrenar, 15 % validar, 15 % probar; nuestras fotos aparte
  para cada foto: encontrar la fruta (máscara) y calcular sus 80 números
  para cada modelo (regresión logística, KNN, SVM, Random Forest, XGBoost):
      probar combinaciones de hiperparámetros con validación cruzada por grupos
  entrenar la red neuronal
  elegir el mejor modelo mirando SOLO los datos de validación
  probarlo UNA vez con los datos de prueba y con nuestras fotos

USAR (la app)
  foto → dejarla de 224×224 → encontrar la fruta → calcular sus 80 números
  calidad ← SVM;  avisar "dudoso" si el modelo no está seguro
  tamaño ← diámetro de la máscara comparado con el de otras frutas del mismo tipo
```

---

## 2. Cómo decidimos resolverlo

### 2.1 Lo que elegimos y por qué

| Decisión | Por qué |
|---|---|
| **Recortar la fruta antes de clasificar** | Así el modelo no puede usar el fondo para hacer trampa. Además, el recorte nos da el área para medir el tamaño. |
| **Repartir las fotos por grupos de toma** | Así las fotos de prueba son realmente nuevas para el modelo y los resultados son honestos. |
| **Describir la fruta con 80 números y usar un SVM** | Con unas 3.900 fotos, resumir lo importante (color, manchas, textura) funciona mejor que pedirle al modelo que lo descubra solo. El SVM es bueno con pocos datos. |
| **Una red neuronal pequeña, entrenada desde cero** | Lo pide el curso, y nos sirve para comparar: la red aprende sola qué mirar. La hicimos pequeña porque no tenemos muchas fotos. |
| **Darle más peso a la fruta *regular*** | Sin eso, el SVM nunca decía "regular". |
| **Medir con macro-F1** | Es la medida que no esconde el fallo en la clase más pequeña. |

### 2.2 Lo que probamos y descartamos

| Alternativa | Por qué la descartamos |
|---|---|
| Repartir las fotos al azar | Quedaban 920 fotos casi iguales entre entrenamiento y prueba, y los resultados salían inflados. |
| Usar la foto completa, con fondo | El modelo aprendía el fondo. La dejamos solo como experimento para comparar. |
| Medir el tamaño contando píxeles "distintos al fondo" | Fallaba con fondos con textura, como los de nuestras fotos. Lo cambiamos por la máscara. |
| Regresión logística | Solo puede trazar fronteras rectas entre clases y se quedó corta (0,60 frente a 0,73 del SVM). |
| Random Forest y XGBoost | Se aprendían los datos de memoria: casi perfectos en entrenamiento, pero mucho peores con datos nuevos. |
| Usar las probabilidades del SVM para elegir la clase | Esas probabilidades se calculan sin tener en cuenta el peso extra de *regular*, y el modelo nunca decía "regular". |
| Transfer learning (aprovechar una red ya entrenada) | El curso lo permite, pero recomienda entrenar desde cero. Lo dejamos como trabajo futuro. |
| Inventar fotos de *regular* (SMOTE) | Con tan pocos ejemplos y una frontera tan difusa, los ejemplos inventados podían salir poco realistas. Preferimos darle más peso a los reales. |

---

## 3. Las matemáticas que usamos

### 3.1 Preparar las fotos

**Color.** Además de RGB usamos el espacio de color **Lab**, que separa la luz ($L^\ast$) del color
($a^\ast$ va de verde a rojo y $b^\ast$ de azul a amarillo). Lo elegimos porque se parece a como vemos las
personas: si dos colores están "cerca" en Lab, también nos parecen parecidos. Así, una sombra cambia sobre
todo $L^\ast$, mientras que una mancha marrón cambia $a^\ast$ y $b^\ast$.

El **tono** (el color puro: rojo, amarillo, verde...) es un ángulo, como en una rueda de colores. Por eso no
se puede promediar normalmente: el promedio de 350° y 10° (dos rojos) daría 180° (cian). Usamos la
**media circular**, que promedia en la rueda:

$$
\bar H = \operatorname{atan2}\Big(\textstyle\sum_i \sin H_i,\ \sum_i \cos H_i\Big)
$$

**Encontrar la fruta: GrabCut** [Rother et al. 2004]. Decide para cada píxel si es fruta o fondo, buscando
la opción que haga más pequeña esta "energía":

$$
E = \underbrace{\sum_{\text{píxeles}} \text{qué tan raro es su color para su grupo}}_{\text{cada píxel en el grupo que le corresponde}}
\;+\; \underbrace{\gamma \sum_{\text{vecinos}} \text{costo de separarlos si se parecen}}_{\text{bordes por donde cambia el color}}
$$

La primera parte pide que cada píxel quede en el grupo cuyo color se le parece. La segunda pide que el
borde entre fruta y fondo pase por donde el color cambia, no por la mitad de una zona pareja. Le dimos una
pista inicial: los bordes de la foto son fondo y el centro con colores vivos es fruta.

**Encontrar fotos repetidas: dHash.** Cada foto se reduce a 256 ceros y unos que resumen si cada píxel es
más claro que el de al lado. Dos fotos son "casi iguales" si sus códigos difieren en muy pocas posiciones
(**distancia de Hamming**). Las fotos casi iguales, o tomadas con segundos de diferencia, forman un mismo
grupo de toma.

**Estandarizar.** Antes de entrenar, cada uno de los 80 números se ajusta así:

$$
z = \frac{x - \text{promedio}}{\text{desviación}}
$$

Sin esto, un número que va de 0 a 100 (como la luz $L^\ast$) pesaría mucho más que uno que va de 0 a 1 (como
una proporción), solo por su escala.

### 3.2 Los 80 números que describen la fruta

Todos se calculan solo con los píxeles de la fruta, nunca con el fondo:

- **Color:** promedio, dispersión y percentiles de cada canal de color.
- **Histogramas:** qué proporción de la fruta tiene cada tono, saturación y brillo.
- **Manchas:** qué proporción de la fruta es mucho más oscura, o de un color muy distinto, que el resto de
  la misma fruta. Al comparar la fruta consigo misma, no importa si la foto se tomó con más o menos luz.
- **Textura:** si la piel es lisa o rugosa, medido con **LBP** [Ojala et al. 2002], que compara cada píxel
  con sus 8 vecinos, y **GLCM** [Haralick et al. 1973], que mide qué tan seguido aparecen juntos tonos de gris
  muy distintos. Una piel sana es lisa; una golpeada o con moho, no.
- **Bordes y forma:** cuántos bordes hay y qué tan redonda o deformada es la fruta.

### 3.3 Los modelos clásicos

**El peso extra para *regular*.** A cada clase le damos un peso según qué tan escasa es:

$$
w_c = \frac{\text{total de fotos}}{3 \times \text{fotos de la clase } c}
\quad\Rightarrow\quad
w_{\text{buena}} = 0{,}65,\ \ w_{\text{regular}} = 8{,}87,\ \ w_{\text{mala}} = 0{,}75
$$

Equivocarse con una fruta *regular* "cuesta" unas 14 veces más que con una buena. Es como si las tres
clases tuvieran la misma cantidad de fotos.

**SVM** [Cortes y Vapnik 1995]. Imagine los 80 números de cada fruta como un punto en un espacio. El SVM
busca la frontera que separa las clases **dejando el mayor espacio libre posible** a cada lado (el
**margen**). Un margen grande hace que el modelo funcione mejor con frutas nuevas. Matemáticamente:

$$
\min\ \underbrace{\tfrac{1}{2}\lVert \mathbf w \rVert^2}_{\text{margen grande}}
\;+\; C \sum_i \underbrace{w_{y_i}\, \xi_i}_{\text{errores, con su peso}}
$$

- $\xi_i$ mide cuánto se pasa la fruta $i$ al lado equivocado.
- $C$ decide qué importa más: un margen amplio o cometer pocos errores.

Como las clases no se pueden separar con una línea recta, usamos el **kernel RBF**:

$$
K(\mathbf x, \mathbf x') = e^{-\gamma \lVert \mathbf x - \mathbf x' \rVert^2}
$$

Mide qué tan parecidas son dos frutas: vale 1 si son idénticas y se acerca a 0 cuando son muy distintas.
Con él, el SVM puede trazar fronteras curvas. Para tres clases se entrenan tres SVM (buena contra regular,
buena contra mala y regular contra mala) y gana la clase con más votos.

Los mejores valores fueron $C = 30$ y $\gamma = 0{,}003$. Un $\gamma$ pequeño significa una frontera suave,
lo que tiene sentido con pocos datos.

**Probabilidades del SVM** [Platt 1999]. El SVM no da probabilidades por sí mismo. scikit-learn las calcula
aparte, con una curva, pero **sin** el peso extra de *regular*. Por eso, en un 8 % de las fotos, la clase
"más probable" no coincide con la que el SVM eligió. La app usa ese desacuerdo como señal de que el
resultado es dudoso.

**Los otros modelos que comparamos:**
- **Regresión logística**: traza fronteras rectas entre las clases.
- **KNN**: mira las $k$ frutas más parecidas del entrenamiento y elige la clase más común entre ellas.
- **Random Forest**: muchos árboles de decisión, cada uno entrenado con una parte de los datos, que votan.
- **XGBoost**: árboles que se entrenan uno tras otro, cada uno corrigiendo los errores del anterior.

**Búsqueda de hiperparámetros.** Para cada modelo probamos varias combinaciones de configuración. Cada una se
evaluó con **validación cruzada en 5 partes**: se entrena con 4 partes y se prueba con la quinta, rotando, y
se promedia el resultado. Las fotos de una misma toma siempre quedan en la misma parte.

### 3.4 La red neuronal (CNN)

La red recibe la foto de la fruta (128 × 128, con el fondo en gris) y la pasa por tres bloques. Cada bloque
hace cuatro cosas:

1. **Convolución.** Pasa pequeños filtros de 3 × 3 píxeles por toda la imagen. Cada filtro aprende a
   detectar algo: un borde, un cambio de color, una mancha.

   $$
   \text{salida}(i,j) = \sum_{\text{canales}} \sum_{u,v} \text{filtro}(u,v) \cdot \text{imagen}(i+u,\ j+v)
   $$

   El mismo filtro se usa en toda la imagen, así que reconoce una mancha esté donde esté.
2. **BatchNorm** [Ioffe y Szegedy 2015]. Reajusta los números para que no se vuelvan demasiado grandes o
   pequeños, lo que hace el entrenamiento más estable.
3. **ReLU**, $\max(0, x)$. Deja pasar los valores positivos y convierte en 0 los negativos. Es lo que
   permite a la red aprender cosas que no son simples sumas.
4. **MaxPool.** Reduce la imagen a la mitad quedándose con el valor más alto de cada cuadro de 2 × 2.

Al final, el **promedio global** resume cada filtro en un solo número, y dos capas densas deciden la clase.
La red tiene unos 102.000 parámetros, muy poco comparado con las redes grandes.

**Cómo aprende.**
- **Softmax** convierte la salida en probabilidades que suman 1.
- La **entropía cruzada** mide qué tan equivocada está la red, de nuevo con el peso extra para *regular*:

  $$
  \text{pérdida} = -\frac{1}{N}\sum_i w_{y_i} \log(\text{probabilidad de la clase correcta})
  $$

- **AdamW** [Loshchilov y Hutter 2019] ajusta los parámetros poco a poco para bajar esa pérdida.
- **Dropout** apaga neuronas al azar mientras entrena, para que no se aprenda los datos de memoria.
- Si la red deja de mejorar, reducimos la velocidad de aprendizaje a la mitad. Si sigue sin mejorar,
  paramos (**early stopping**).
- **Aumento de datos.** En cada vuelta le mostramos las fotos con pequeños cambios (giradas, más claras,
  más grandes) para que aprenda que esos cambios no alteran la calidad.

### 3.5 El tamaño

El área de la fruta es la proporción de píxeles de la máscara que son fruta. Para pasarla a un "diámetro",
imaginamos un círculo con esa misma área:

$$
\pi \left(\frac{d}{2}\right)^2 = \text{área} \quad\Rightarrow\quad d_{eq} = 2\sqrt{\frac{\text{área}}{\pi}}
$$

Este diámetro equivalente no depende de si la fruta está girada. Después dividimos las frutas de
entrenamiento de cada tipo en **tres grupos iguales** (terciles): el tercio de diámetros más pequeños es
"pequeño", el del medio "mediano" y el de arriba "grande".

### 3.6 Cómo medimos los resultados

Además del macro-F1 usamos:
- **Exactitud balanceada:** el promedio del recall de las tres clases.
- **Kappa de Cohen** [Cohen 1960]: cuánto acierta el modelo **por encima de lo que acertaría adivinando**.
  Vale 0 si no es mejor que adivinar y 1 si es perfecto.

  $$
  \kappa = \frac{\text{acierto real} - \text{acierto por azar}}{1 - \text{acierto por azar}}
  $$

- **Márgenes de error (bootstrap)** [Efron y Tibshirani 1993]. Con solo 46 fotos propias, un resultado
  puede deberse a la suerte. Para saber cuánto confiar en él:
  1. sacamos al azar 46 fotos de las mismas 46, pudiendo repetir;
  2. calculamos el resultado;
  3. lo repetimos miles de veces.

  El rango donde cae el 95 % de los resultados es el **intervalo de confianza**. Para comparar dos modelos
  hacemos lo mismo con la *diferencia* entre ellos: si el intervalo no incluye el 0, la diferencia es real.

---

## 4. Qué nos dicen los resultados

### 4.1 Los números (macro-F1)

| Modelo | Validación cruzada | Validación | Prueba (Kaggle) | Nuestras fotos [margen 95 %] |
|---|---|---|---|---|
| Adivinar al azar | — | 0,31 | 0,33 | 0,35 |
| Regresión logística | 0,60 | 0,60 | — | — |
| KNN | 0,65 | 0,65 | — | — |
| Random Forest | 0,68 | 0,63 | — | — |
| XGBoost | 0,69 | 0,67 | — | — |
| **SVM** | **0,72** | **0,73** | **0,72** | **0,47** [0,34 a 0,58] |
| Red neuronal | — | 0,65 | 0,65 | 0,38 [0,29 a 0,47] |

Solo probamos con los datos de prueba los modelos elegidos en validación: el SVM y la red neuronal. Las
gráficas están en `docs/figures/`.

### 4.2 Qué aprendimos de ellos

- **Con fotos de Kaggle, el SVM es estable.** Da casi lo mismo en validación cruzada (0,72), en validación
  (0,73) y en prueba (0,72). Eso indica que no tuvimos suerte al elegirlo y que funciona igual con fotos que
  nunca vio, siempre que sean parecidas.
- **Algunos modelos se aprendieron los datos de memoria (sobreajuste).** Se nota al comparar cómo les va con
  los datos de entrenamiento y con datos nuevos:

  | Modelo | Entrenamiento | Datos nuevos | Diferencia | Qué significa |
  |---|---|---|---|---|
  | XGBoost | 0,99 | 0,69 | 0,30 | Memorizó |
  | Random Forest | 0,94 | 0,68 | 0,26 | Memorizó |
  | SVM | 0,86 | 0,72 | 0,14 | Buen equilibrio |
  | Regresión logística | 0,66 | 0,60 | 0,06 | Se quedó corta (subajuste) |

  El SVM aprende lo suficiente sin memorizar, gracias a su margen y a su frontera suave.
- **La frontera entre clases es curva.** El SVM con kernel RBF le gana por 0,12 a la regresión logística,
  que solo traza rectas.
- **El peso extra para *regular* es indispensable.** Sin él, el SVM nunca dice "regular" y su macro-F1 baja
  a 0,63, aunque otras medidas como el kappa parezcan mejorar (0,86). Por eso usamos macro-F1: no esconde el
  problema.
- **El SVM le gana a la red neuronal.** En prueba, por 0,076, y el margen de error confirma que la
  diferencia es real. Con pocas fotos, describir la fruta con conocimiento (color en Lab, manchas relativas)
  funcionó mejor que dejar que la red aprendiera sola qué mirar. La red encuentra más frutas *regular*, pero
  se equivoca mucho al hacerlo: de cada 7 que llama "regular", solo una lo es.
- **Con nuestras fotos, todo cae.** El SVM pasa de 0,72 a 0,47. Esa caída es más grande que cualquier
  diferencia entre modelos, así que el problema no es el modelo sino lo distintas que son las fotos. Las
  causas que medimos:
  - en solo 7 de 46 fotos la máscara quedó limpia; en las demás se coló una sombra o un pedazo de pared;
  - la fruta ocupa menos espacio en la foto;
  - tenemos otra idea de madurez: en Kaggle, el 93 % de las naranjas verdes son "buenas", y para nosotros
    eran "regular".
- **El fondo sí influía, pero no tanto.** Con nuestras fotos, recortar la fruta mejoró los resultados en
  los tres casos que probamos (+0,02, +0,08 y +0,13), pero solo el último es claramente distinto de la
  suerte. El efecto se notó en la red neuronal, no en el SVM.
- **El tamaño se comporta como predice la fórmula.** En nuestras fotos, 38 de 46 frutas salieron
  "pequeñas" porque las tomamos más lejos, no porque lo fueran. El diámetro en la foto baja cuando la
  cámara se aleja.

### 4.3 Comparación con otros trabajos

Los trabajos que revisan este tema [Bhargava y Bansal 2021; Naik y Patel 2017] reportan aciertos altos, a
menudo por encima del 90 %, con fotos tomadas en condiciones controladas. Con fotos de Kaggle, nosotros
también separamos bien la fruta buena de la mala (F1 de 0,89 y 0,95).

Lo que nos pasó con el fondo y con nuestras fotos tiene nombre en la literatura:
- **aprendizaje por atajos** (*shortcut learning*) [Geirhos et al. 2020]: el modelo aprende una pista fácil,
  como el fondo, en vez de lo que de verdad importa;
- **fuga de datos** (*data leakage*) [Kaufman et al. 2012]: las fotos repetidas inflan los resultados.

Pocos trabajos incluyen una clase intermedia como *regular* o prueban con fotos de otro origen. En eso se
diferencia este proyecto.

---

## 5. Conclusiones

1. **Dividir el problema en pasos fue una buena decisión.** Separar primero la fruta nos permitió evitar la
   trampa del fondo, medir el tamaño y entender los errores: la mayoría de los fallos con nuestras fotos se
   explican por una máscara mal hecha.
2. **La teoría se cumplió.** El SVM, con su margen amplio, generalizó mejor que los árboles, que memorizan
   cuando hay pocos datos. El peso extra fue indispensable para la clase pequeña. Y el tamaño depende de la
   distancia, tal como dice su fórmula.
3. **No alcanzamos la meta con nuestras fotos** (0,47 frente a 0,70). La razón no es el modelo sino que
   nuestras fotos son muy distintas a las de Kaggle.
4. **Cómo lo mejoraríamos:**
   1. **Mejor recorte de la fruta:** que GrabCut ignore los bordes de relleno y no confunda sombras con
      fruta.
   2. **Más fotos nuestras,** con la misma luz, el mismo fondo y la misma distancia, y otras fotos nuevas
      para probar (las 46 actuales ya las usamos).
   3. **Un criterio de *regular* acordado** entre varias personas, midiendo con kappa qué tanto coinciden.
   4. **Detectar manchas pequeñas** con descriptores por zonas, o usar transfer learning.
   5. **Que el modelo diga cuándo una foto es muy distinta** a las que conoce.
   6. **Medir el tamaño en centímetros** poniendo al lado un objeto de tamaño conocido, por ejemplo una
      moneda:

      $$
      \text{diámetro (cm)} = d_{eq} \times \frac{\text{diámetro real de la moneda}}{d_{eq} \text{ de la moneda}}
      $$
