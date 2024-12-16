# Prediction
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet
from tensorflow.keras.models import load_model
from src.scripts.data import image_processing
import numpy as np

# Display
import streamlit as st
from PIL import Image 

@st.cache_resource
def load_model_and_class_names():
    model = load_model("model_resnet_dog_breeds.h5", compile=False)
    class_names = ['Chihuahua', 'Japanese spaniel', 'Maltese dog', 'Pekinese', 'Shih-Tzu']
    return model, class_names

def load_image(img):
    img = np.array(img)
    img = image_processing.resize_image(img)
    # img = image_processing.equalize_image(img)
    # img = image_processing.whiten_image(img)
    # img = image_processing.normalize_image(img)
    img = preprocess_resnet(img)
    img = np.expand_dims(img, axis=0)
    return img

def predict_race(model, img, class_names):
    preds = model.predict(img)
    pred_idx = np.argmax(preds)
    pred_label = class_names[pred_idx]
    probabilities = {class_names[i]: preds[0][i] for i in range(len(class_names))}
    
    return pred_label, probabilities

def main():

    st.set_page_config(
    page_title="Détecteur de race de chien",
    page_icon="🐶",
    layout="wide",  # Garder le layout large
    initial_sidebar_state="expanded"
)
    
    # Chargement du modèle et des classes
    model, class_names = load_model_and_class_names()

    # Titre et description
    st.title("Détecteur de race de chien")
    st.write("Ce modèle détecte la race de chien à partir d'une image.")

    # Upload de fichier
    file = st.file_uploader("Veuillez charger une image (formats acceptés : JPG, PNG)", type=["jpg", "png"])

    # Placeholder pour l'image et les résultats
    col1, col2 = st.columns(2)
    img_placeholder = col1.empty()
    res_placeholder = col2.empty()

    
    # Si une image est chargée
    if file is not None:
        try:
            with st.spinner("Chargement de l'image..."):
                img = Image.open(file)
                col1, col2 = st.columns([1, 2])

                with col1:
                    img_placeholder = st.image(img, caption="Image chargée", use_container_width=True)

                with col2:
                    pass

            if st.button("Lancer la détection de race"):
                with st.spinner("Analyse en cours..."):
                    img_tensor = load_image(img)
                    prediction, probabilities = predict_race(model, img_tensor, class_names)

                    with col2:
                        st.success(f"Race prédite : **{prediction}**")
                        st.write("Probabilités des principales races détectées :")
                        for class_name, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:5]:  # Top 5
                            st.write(f"{class_name} :")
                            st.progress(float(prob)) 
        except Exception as e:
            st.error(f"Erreur lors du traitement de l'image : {e}")

if __name__ == "__main__":
    main()