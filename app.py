import streamlit as st
import matplotlib.pyplot as plt

import keras_ocr
import pandas as pd
from rake_nltk import Rake

import nltk
nltk.download('stopwords')
nltk.download('punkt')
from PIL import Image
import webbrowser


import requests
from sentence_transformers import SentenceTransformer


st.set_page_config(page_title="OCR with Streamlit", page_icon=":camera:", layout="wide")


pipeline = keras_ocr.pipeline.Pipeline()

def keras_ocr_process(img):

    images = [
            keras_ocr.tools.read(img)
        ]

    prediction_groups = pipeline.recognize(images)

    testo = []

    for elem in prediction_groups[0]:
        testo.append(elem[0])

    testo_unito = ' '.join(testo)
    rake_nltk_var = Rake()
    rake_nltk_var.extract_keywords_from_text(testo_unito)
    keyword_extracted = rake_nltk_var.get_ranked_phrases()
    return keyword_extracted[0]


df_product = pd.read_xml('https://volteco.com/product-sitemap.xml')
list_of_product = df_product['loc'].apply( lambda x: x.rsplit('/', 1)[-1])
df_product['name'] = list_of_product
#df_product


model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

API_TOKEN = "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"

API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()
    

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with st.spinner('Stiamo processando la foto...'):
        output = query({
            "inputs": {
                "source_sentence": keras_ocr_process(uploaded_file),
                "sentences": df_product["name"].values.tolist()
            },
        })

    pos_max = output.index(max(output))
    st.write(f"Ehi ehi, hai davanti a te un bel {df_product.iloc[pos_max]['name']}")

    if st.button('Vai al prodotto'):
        webbrowser.open_new_tab(df_product.iloc[pos_max]['loc'])