import streamlit as st
import requests
from PIL import Image
from io import BytesIO

# Your Unsplash API key
UNSPLASH_ACCESS_KEY = '18wgpNbjDKI3LKrXKfoVnb4oreoajorKWof1zt3oWMk'

def fetch_image_from_unsplash(query):
    url = f"https://api.unsplash.com/photos/random?query={query}&client_id={UNSPLASH_ACCESS_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        image_url = data['urls']['regular']
        return fetch_image_from_url(image_url)
    else:
        return None

def fetch_image_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    else:
        return None

st.title("Text-to-Image Generation")
text_prompt = st.text_input("Enter a textual description:")
if text_prompt:
    generated_image = fetch_image_from_unsplash(text_prompt)
    if generated_image:
        st.image(generated_image, caption=text_prompt)
    else:
        st.write("Failed to fetch the image.")
