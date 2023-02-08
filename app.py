import streamlit as st

import easyocr
import pandas as pd
from rake_nltk import Rake

import nltk
nltk.download('stopwords')
nltk.download('punkt')
from PIL import Image
import numpy as np

import requests
import webbrowser
import cv2
import skimage.measure
from skimage.transform import rescale, resize, downscale_local_mean



st.set_page_config(page_title="OCR with Streamlit", page_icon=":camera:", layout="wide")

st.title("VOLTECO Product Recognizer")

def easy_ocr_process(img):

    reader = easyocr.Reader(['it','en']) # this needs to run only once to load the model into memory

    image = Image.open(img)

    height, width = image.size
    if width > 1000:
        image = image.resize((height//2 ,width//2), Image.ANTIALIAS)
 
    testo = reader.readtext(np.asarray(image), detail = 0, paragraph=True)
    testo_clean = ' '.join(testo)
    st.write(f"Testo rilevato: {testo_clean.lower()}")

    rake_nltk_var = Rake()
    rake_nltk_var.extract_keywords_from_text(testo_clean)
    keyword_extracted = rake_nltk_var.get_ranked_phrases()
    return keyword_extracted[0]


df_product = pd.read_xml('https://volteco.com/product-sitemap.xml')
list_of_product = df_product['loc'].apply( lambda x: x.rsplit('/', 1)[-1])
df_product['name'] = list_of_product
#df_product


API_TOKEN = "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"

API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def find_top_3_indices(numbers):
    sorted_numbers = sorted(enumerate(numbers), key=lambda x: x[1], reverse=True)
    return [i[0] for i in sorted_numbers[:3]], [i[1] for i in sorted_numbers[:3]]
    

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with st.spinner('Stiamo processando la foto...'):
        output = query({
            "inputs": {
                "source_sentence": easy_ocr_process(uploaded_file),
                "sentences": df_product["name"].values.tolist()
            },
        })

    best_three, probability = find_top_3_indices(output)
    for i in range(3):
        st.write(f"Ehi ehi, potresti avere davanti: **{df_product.iloc[best_three[i]]['name']}**, con una probabilità del **{probability[i]*100:.2f}%**")
        if st.button(f"Vai al prodotto {i+1}: {df_product.iloc[best_three[i]]['name']}"):
            webbrowser.open_new_tab(df_product.iloc[best_three[i]]['loc'])
       


