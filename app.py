import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from PIL import Image
import random
from streamlit_drawable_canvas import st_canvas

# Charger le modèle
filepath = 'C:/Users/ryanh/Downloads/CNN/best_model.keras'
model = load_model(filepath)

# Fonction de reformatage des données du CSV
def reformat(df, nb):
    df_np = df.astype(float).to_numpy().reshape((-1, 28, 28))
    image = np.expand_dims(df_np[nb], axis=-1)  # Ajouter un canal
    image = image / 255.0  # Normaliser les valeurs de l'image
    image = np.expand_dims(image, axis=0)  # Ajouter la dimension batch
    return image

# Fonction de prédiction
def predict(image):
    if isinstance(image, Image.Image):
        image = image.resize((28, 28)).convert('L')
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=[0, -1])
    prediction = model.predict(image)
    return np.argmax(prediction), np.max(prediction)

# Fonction pour dessiner une image
def draw_canvas():
    st.markdown("### Dessinez un chiffre")
    canvas_result = st_canvas(
        fill_color="#000000",
        stroke_width=10,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=200,
        width=200,
        drawing_mode="freedraw",
        key="canvas",
    )
    if canvas_result.image_data is not None:
        image = Image.fromarray(np.uint8(canvas_result.image_data))
        return image
    return None

# Fonction pour afficher des boutons de feedback avec couleur
def display_feedback_buttons():
    col1, col2 = st.columns([1, 1])  # Crée deux colonnes de largeur égale

    with col1:
        if st.button('Bonne réponse', key='good_response'):
            st.write("Merci pour votre retour !")
            # Enregistrer le feedback ici, par exemple dans un fichier CSV ou une base de données

    with col2:
        if st.button('Mauvaise réponse', key='bad_response'):
            st.write("Désolé, nous deviendrons meilleurs !")
            # Enregistrer le feedback ici, par exemple dans un fichier CSV ou une base de données

    # CSS pour styliser les boutons
    st.markdown("""
        <style>
        .stButton button[data-testid="stButton"] {
            border-radius: 5px;
            padding: 10px 20px;
            font-size: 16px;
            margin: 5px;
            cursor: pointer;
            border: none;
        }
        .stButton button[data-testid="stButton"][key="good_response"] {
            background-color: #4CAF50; /* Vert */
            color: white;
        }
        .stButton button[data-testid="stButton"][key="bad_response"] {
            background-color: #f44336; /* Rouge */
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)

# Interface utilisateur Streamlit
st.title('Prédiction de chiffre')

# Options d'entrée
option = st.selectbox(
    'Choisissez le type d\'entrée',
    ('Dessiner un chiffre', 'Télécharger un fichier CSV')
)

if option == 'Dessiner un chiffre':
    image = draw_canvas()
    if image:
        st.image(image, caption='Image dessinée.', use_column_width=True)
        st.write("Classifiant...")
        label, confidence = predict(image)
        st.write(f"Prédiction : {label} avec une confiance de {confidence:.2f}")

        # Afficher les boutons de feedback
        display_feedback_buttons()

elif option == 'Télécharger un fichier CSV':
    uploaded_file = st.file_uploader("Choisissez un fichier CSV...", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(df)

        if st.button('Choisir un chiffre au hasard'):
            nb = random.randint(0, len(df) - 1)
            image = reformat(df, nb)

            st.session_state['image'] = image
            st.session_state['nb'] = nb

        if 'image' in st.session_state:
            st.image(st.session_state['image'][0].reshape(28, 28), caption=f'Image reformatée index {st.session_state["nb"]}', use_column_width=True)
            if st.button('Lancer la prédiction'):
                prediction = model.predict(st.session_state['image'])
                label = np.argmax(prediction)
                confidence = np.max(prediction)
                st.write(f"Prédiction : {label} avec une confiance de {confidence:.2f}")

                # Afficher les boutons de feedback
                display_feedback_buttons()
