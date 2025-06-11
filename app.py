import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
from fpdf import FPDF
from io import BytesIO
import datetime

# Configuración de la página
st.set_page_config(
    page_title="Detector de Neumonía por IA",
    page_icon="🫁",
    layout="centered"
)

# Cargar modelo
model = load_model("modelo_neumonia_mobilenetv2.h5")

# Función para preprocesar la imagen
def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    image = img_to_array(image) / 255.0
    return np.expand_dims(image, axis=0)

# Función para generar informe en PDF con análisis detallado
def generar_informe(confianza):
    pdf = FPDF()
    pdf.add_page()

    # --- Encabezado con logos ---
    pdf.image("C:/Users/SAIMJ/Desktop/IA_Pneumonia/pdf/uncp_logo.png", 10, 8, 33)    # Logo UNCP
    pdf.image("C:/Users/SAIMJ/Desktop/IA_Pneumonia/pdf/fis_uncp_logo.png", 170, 8, 33) # Logo FIS

    pdf.set_font("Arial", 'B', 14)
    pdf.set_xy(0, 15)
    pdf.cell(210, 10, "Universidad Nacional del Centro del Perú", 0, 1, 'C')

    pdf.ln(15)

    # --- Cuerpo del informe ---
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Informe IA - Detección de Neumonía", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Fecha: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}", ln=True)
    pdf.ln(5)

    texto = f"""
Análisis detallado para evaluación médica profesional:

- Confianza del modelo en la detección: {confianza:.2%}

- Descripción técnica:
  El modelo, basado en MobileNetV2 con transferencia de aprendizaje,
  ha identificado patrones característicos en la imagen de rayos X,
  tales como opacidades en parénquima pulmonar, que pueden indicar
  la presencia de neumonía.

- Recomendaciones:
  Se sugiere realizar una evaluación clínica completa, complementada
  con otros exámenes diagnósticos para confirmar el diagnóstico y definir
  el tratamiento adecuado.

IMPORTANTE: Este informe es una ayuda para profesionales de la salud y
no reemplaza el juicio clínico ni el diagnóstico médico presencial.
    """

    pdf.multi_cell(0, 10, texto.strip())

    pdf_output = pdf.output(dest='S').encode('latin1')
    return BytesIO(pdf_output)

# Encabezado de la app
st.markdown("<h1 style='text-align: center; color: #2E86C1;'>🫁 Detector de Neumonía</h1>", unsafe_allow_html=True)
st.write("Sube una imagen de rayos X y deja que la inteligencia artificial te diga si detecta neumonía.")

# Subida de imagen
uploaded_file = st.file_uploader("📄 Sube una imagen (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='🖼 Imagen subida', use_container_width=True)
    st.markdown("---")

    with st.spinner('Analizando imagen...'):
        img = preprocess_image(image)
        prediction = model.predict(img)[0][0]

    confidence = prediction if prediction > 0.5 else 1 - prediction

    st.markdown("### Resultado del modelo:")
    if prediction > 0.5:
        st.error(f"🔴 *Posible Neumonía Detectada* — Confianza: **{confidence:.2%}**")
        st.warning("⚠️ Esta imagen presenta características que podrían indicar neumonía. Se recomienda acudir a un centro de salud.")

        st.markdown("#### 🧪 Interpretación sencilla:")
        st.markdown("""
- La IA ha analizado zonas de la radiografía que suelen verse alteradas por una posible infección pulmonar.
- Esto **no reemplaza** una revisión médica, pero puede ayudarte a detectar señales tempranas.
- Se recomienda realizar una revisión médica complementaria con estos resultados.
        """)

        # Generar PDF en memoria con análisis detallado
        pdf_buffer = generar_informe(confidence)

        st.download_button(
            label="📄 Descargar informe en PDF",
            data=pdf_buffer,
            file_name="informe_neumonia.pdf",
            mime="application/pdf"
        )
    else:
        st.success(f"🟢 *Paciente Sano* — Confianza: **{confidence:.2%}**")

    st.progress(int(confidence * 100))
    st.info("El modelo se basa en MobileNetV2, entrenado con transferencia de aprendizaje para identificar neumonía.")

# Footer
st.markdown("""
---
<p style="text-align: center; font-size: 14px;">
  Creado por <strong>Saim Jesús Veliz Carrasco</strong> | UNCP 2025 | Proyecto de detección de neumonía con IA 🧠
</p>
""", unsafe_allow_html=True)
