"""App de clasificación de calidad y tamaño de fruta (Streamlit).

    streamlit run app/app.py

Flujo (idéntico al de entrenamiento y evaluación, src/evaluation/prediccion.py):
    foto (cámara o archivo) → EXIF + RGB + cuadrar con relleno + 224×224
    → segmentación GrabCut (máscara de la fruta)
    → SVM sobre 80 características de la fruta (modelo recomendado en el notebook 05)
      o CNN sobre la fruta con fondo gris (opcional, si existe el checkpoint)
    → calidad (good / regular / bad) + tamaño (d_eq y tercil por fruta, cortes de train).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.config import CHECKPOINTS_DIR, FRUITS, QUALITY_CLASSES, SIZE_CLASSES, SPLITS_CSV  # noqa: E402
from src.data.segmentacion import aplicar_mascara, segmentar_fruta  # noqa: E402
from src.evaluation.prediccion import diametro_equivalente, preprocesar  # noqa: E402

FRUTA_ES = {"apple": "Manzana", "banana": "Banano", "orange": "Naranja"}
CALIDAD_ES = {"good": "Buena", "regular": "Regular", "bad": "Mala"}
CALIDAD_COLOR = {"good": "#2e7d32", "regular": "#ef6c00", "bad": "#c62828"}
CALIDAD_TEXTO = {
    "good": "Sin defectos visibles: apta para venta.",
    "regular": "Defectos leves o madurez fuera de punto: venta rápida o procesamiento.",
    "bad": "Daño o podredumbre: descartar o separar.",
}
TAMANO_ES = {"pequeno": "Pequeño", "mediano": "Mediano", "grande": "Grande"}
MODELOS = {
    "SVM (recomendado)": "svm",
    "CNN desde cero": "cnn",
}

st.set_page_config(page_title="Calidad de fruta", page_icon="🍎", layout="wide")


# ----------------------------------------------------------------- recursos
@st.cache_resource(show_spinner="Cargando el modelo...")
def cargar_modelo(tipo: str):
    if tipo == "svm":
        from src.evaluation.prediccion import ModeloSVM
        return ModeloSVM()
    from src.evaluation.prediccion import ModeloCNN
    return ModeloCNN()


@st.cache_data
def referencia_train() -> pd.DataFrame:
    """Cortes de tamaño (terciles de d_eq) y rango de área de la máscara por fruta, SOLO con train."""
    df = pd.read_csv(SPLITS_CSV)
    tr = df[df.split == "train"]
    filas = []
    for fruta, g in tr.groupby("fruit"):
        c1, c2 = np.percentile(g.d_eq, [100 / 3, 200 / 3])
        a1, a99 = np.percentile(g.mask_area, [1, 99])
        filas.append(dict(fruta=fruta, corte_1=c1, corte_2=c2, area_p1=a1, area_p99=a99))
    return pd.DataFrame(filas).set_index("fruta")


def modelos_disponibles() -> dict[str, str]:
    archivos = {"svm": CHECKPOINTS_DIR / "ml_mejor.joblib", "cnn": None}
    try:
        from src.training.entrenar_cnn import PRINCIPAL
        archivos["cnn"] = CHECKPOINTS_DIR / f"cnn_{PRINCIPAL}.pt"
    except Exception:  # noqa: BLE001 (torch no instalado, etc.)
        pass
    return {n: t for n, t in MODELOS.items() if archivos[t] is not None and archivos[t].exists()}


# ----------------------------------------------------------------- lógica
def tamano(d_eq: float, fruta: str, ref: pd.DataFrame) -> str:
    c1, c2 = ref.loc[fruta, ["corte_1", "corte_2"]]
    return SIZE_CLASSES[0] if d_eq <= c1 else (SIZE_CLASSES[1] if d_eq <= c2 else SIZE_CLASSES[2])


def avisos_segmentacion(mascara: np.ndarray, fruta: str, ref: pd.DataFrame) -> list[str]:
    """Señales simples de que la máscara (y por tanto la predicción) puede no ser fiable."""
    avisos = []
    area = float(mascara.mean())
    b = max(2, int(0.03 * mascara.shape[0]))
    borde = np.concatenate([mascara[:b].ravel(), mascara[-b:].ravel(), mascara[:, :b].ravel(), mascara[:, -b:].ravel()])
    if area < 0.02:
        avisos.append("No se encontró bien la fruta (máscara casi vacía).")
    elif area < ref.loc[fruta, "area_p1"]:
        avisos.append("La fruta ocupa muy poco de la imagen: acerque la cámara a la distancia indicada.")
    if area > ref.loc[fruta, "area_p99"]:
        avisos.append("La máscara es más grande de lo normal: puede incluir fondo o sombra.")
    if borde.mean() > 0.15:
        avisos.append("La máscara toca el borde de la imagen: es probable que haya tomado parte del fondo.")
    return avisos


def superponer(rgb: np.ndarray, mascara: np.ndarray) -> np.ndarray:
    """Fruta en color y el resto oscurecido, con el contorno de la máscara en verde."""
    import cv2
    out = (rgb * 0.35).astype(np.uint8)
    out[mascara] = rgb[mascara]
    contornos, _ = cv2.findContours(mascara.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, contornos, -1, (0, 255, 0), 2)
    return out


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


def predecir_svm(modelo, rgb: np.ndarray, mascara: np.ndarray):
    """Clase con predict() (la evaluada) y barras con la función de decisión del SVM.

    Las probabilidades de Platt (predict_proba) se ajustan aparte y sin los pesos de clase, así que
    en ~8 % de las fotos su máximo no coincide con predict() (74 de 917 en val). La función de
    decisión es la que usa el propio SVM para elegir la clase: su máximo coincide con predict() en
    el 99,6 % de val. Se muestra normalizada con softmax como puntaje relativo, no como probabilidad.
    Si Platt y predict() no coinciden, la foto queda cerca de la frontera: se marca como dudosa.
    """
    from src.features.extraccion import extraer
    x = extraer(rgb, mascara)[None]
    pipe = modelo.modelo
    clase = int(pipe.predict(x)[0])
    puntaje = softmax(pipe.decision_function(x)[0])
    platt = pipe.predict_proba(x)[0]
    dudoso = int(np.argmax(platt)) != clase or int(np.argmax(puntaje)) != clase
    return clase, puntaje, dudoso


def analizar(img: Image.Image, tipo: str):
    t0 = time.perf_counter()
    rgb = preprocesar(img)
    mascara = segmentar_fruta(rgb)
    t1 = time.perf_counter()
    modelo = cargar_modelo(tipo)
    if tipo == "svm":
        clase, puntaje, dudoso = predecir_svm(modelo, rgb, mascara)
    else:
        clase, puntaje = modelo.predecir(rgb, mascara)
        dudoso = False
    t2 = time.perf_counter()
    return rgb, mascara, clase, puntaje, dudoso, (t1 - t0) * 1000, (t2 - t1) * 1000


# ----------------------------------------------------------------- interfaz
st.title("Clasificación de calidad y tamaño de fruta")
st.caption("Proyecto final · Algoritmos y Programación III · Universidad Icesi")

disponibles = modelos_disponibles()
with st.sidebar:
    st.header("Configuración")
    fruta = st.selectbox("Fruta", FRUITS, format_func=FRUTA_ES.get,
                         help="Se usa para los cortes de tamaño, que son distintos por fruta.")
    if not disponibles:
        st.error("No hay modelos entrenados en experiments/checkpoints/. Ver docs/instalacion.md.")
        st.stop()
    nombre_modelo = st.radio("Modelo", list(disponibles))
    tipo = disponibles[nombre_modelo]
    st.divider()
    st.subheader("Cómo tomar la foto")
    st.markdown(
        "- **Una sola fruta**, centrada y completa.\n"
        "- **Fondo liso** y de color distinto a la fruta.\n"
        "- **Luz pareja**, sin sombras fuertes ni reflejos.\n"
        "- **Distancia fija** a la cámara: el tamaño se mide en la imagen, "
        "así que cambia si la cámara se acerca o se aleja."
    )
    st.divider()
    st.caption(
        "Rendimiento medido (macro-F1, notebook 05): SVM 0,72 en fotos tipo Kaggle y 0,47 en nuestras fotos; "
        "la clase *regular* casi no se detecta en fotos propias. Úsese como **apoyo**, no como decisión final."
    )

pestana_camara, pestana_archivo = st.tabs(["📷 Cámara", "📁 Subir imagen"])
with pestana_camara:
    foto_camara = st.camera_input("Tome una foto de la fruta")
with pestana_archivo:
    foto_archivo = st.file_uploader("Imagen JPG, PNG o HEIC", type=["jpg", "jpeg", "png", "webp", "heic", "heif"])

foto = foto_camara or foto_archivo
if foto is None:
    st.info("Tome una foto o suba una imagen para analizarla.")
    st.stop()

try:
    if foto.name.lower().endswith((".heic", ".heif")):
        from pillow_heif import register_heif_opener
        register_heif_opener()
    imagen = Image.open(foto)
except Exception as e:  # noqa: BLE001
    st.error(f"No se pudo abrir la imagen: {e}")
    st.stop()

with st.spinner("Segmentando y clasificando..."):
    rgb, mascara, clase, puntaje, dudoso, ms_seg, ms_mod = analizar(imagen, tipo)

ref = referencia_train()
calidad = QUALITY_CLASSES[clase]
d_eq = diametro_equivalente(mascara)
tam = tamano(d_eq, fruta, ref)

col_img, col_res = st.columns([3, 2], gap="large")
with col_img:
    c1, c2, c3 = st.columns(3)
    c1.image(rgb, caption="Entrada (224×224)", width="stretch")
    c2.image(superponer(rgb, mascara), caption="Fruta detectada", width="stretch")
    c3.image(aplicar_mascara(rgb, mascara), caption="Lo que ve el modelo", width="stretch")
    for aviso in avisos_segmentacion(mascara, fruta, ref):
        st.warning(aviso)

with col_res:
    st.markdown(
        f"<div style='padding:1rem 1.2rem;border-radius:0.6rem;border-left:8px solid {CALIDAD_COLOR[calidad]};"
        f"background:rgba(128,128,128,0.08)'>"
        f"<div style='font-size:0.9rem;opacity:0.75'>Calidad · {FRUTA_ES[fruta]}</div>"
        f"<div style='font-size:2.2rem;font-weight:700;color:{CALIDAD_COLOR[calidad]}'>{CALIDAD_ES[calidad]}</div>"
        f"<div>{CALIDAD_TEXTO[calidad]}</div></div>",
        unsafe_allow_html=True,
    )
    st.write("")

    if dudoso:
        st.warning(
            "**Resultado dudoso.** La fruta quedó cerca de la frontera entre dos clases: dos formas de medir "
            "la confianza del modelo no coinciden. Conviene revisarla a mano o repetir la foto con mejor luz y fondo."
        )

    if tipo == "svm":
        st.markdown("**Puntaje del modelo por clase**",
                    help="Función de decisión del SVM normalizada (softmax). Es la medida con la que el SVM "
                         "elige la clase, pero no es una probabilidad calibrada.")
    else:
        st.markdown("**Probabilidad por clase** (softmax de la CNN)")
    for i, c in enumerate(QUALITY_CLASSES):
        st.progress(float(puntaje[i]), text=f"{CALIDAD_ES[c]}: {puntaje[i]:.0%}")

    st.markdown("**Tamaño**")
    m1, m2 = st.columns(2)
    m1.metric("Categoría", TAMANO_ES[tam])
    m2.metric("Diámetro equivalente", f"{d_eq:.3f}", help="d_eq = 2·√(área de la máscara / π), "
              "como fracción del lado de la imagen.")
    c_1, c_2 = ref.loc[fruta, ["corte_1", "corte_2"]]
    st.caption(f"Cortes para {FRUTA_ES[fruta].lower()} (terciles de train): pequeño ≤ {c_1:.3f} < mediano ≤ {c_2:.3f} "
               "< grande. Válido solo a la distancia de captura de referencia.")

    st.caption(f"Tiempo: segmentación {ms_seg:.0f} ms · modelo {ms_mod:.0f} ms · {nombre_modelo}")
