import av
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
from ultralytics import YOLO

st.set_page_config(page_title="Detección en Tiempo Real con YOLO", layout="centered")
st.title("🛸 Detección de Objetos en Tiempo Real con YOLO")
st.write("Permite el acceso a tu cámara para iniciar la detección de objetos con YOLO26.")


@st.cache_resource
def cargar_modelo():
    return YOLO("yolo26n.pt")


model = cargar_modelo()
umbral_confianza = st.sidebar.slider("Umbral de confianza", 0.05, 1.0, 0.50, 0.05)


class YOLOVideoProcessor(VideoProcessorBase):
    def __init__(self, modelo, conf: float):
        self.model = modelo
        self.conf = conf

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        results = self.model(img, conf=self.conf, imgsz=640, verbose=False)
        anotado = results[0].plot()
        return av.VideoFrame.from_ndarray(anotado, format="bgr24")


ctx = webrtc_streamer(
    key="yolo-detection",
    video_processor_factory=lambda: YOLOVideoProcessor(model, umbral_confianza),
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
)

# Actualiza el umbral en vivo si el streamer ya está activo
if ctx.video_processor is not None:
    ctx.video_processor.conf = umbral_confianza