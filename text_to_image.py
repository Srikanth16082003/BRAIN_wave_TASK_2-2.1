from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import requests
from io import BytesIO

# Load the model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
def fetch_image_from_url(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content))

def get_image(text):
    # Fetch a sample image (this would ideally be replaced with a proper generation process)
    url = "https://via.placeholder.com/512"
    image = fetch_image_from_url(url)
    return image
import matplotlib.pyplot as plt

def show_image(image):
    plt.imshow(image)
    plt.axis('off')
    plt.show()

# Example usage
text_prompt = "A beautiful sunset over a mountain range"
generated_image = get_image(text_prompt)
show_image(generated_image)
