import clip
import torch

from numpy import ndarray
from typing import List
from PIL import Image


# Build a class to extract embeddings of images and texts using CLIP
class CLIP:
    def __init__(self, model_name: str="ViT-B/32", device: str = "cpu"):
        
        self.device = device
        self.model_name = model_name

        # Load the model
        self.model, self.preprocess = clip.load(self.model_name, device=self.device)

    def get_image_embedding(self, image_files: List[str]) -> ndarray:
        """
        Get the embeddings of a list of images
        """
        list_image_embeddings = []

        for image_path in image_files:
            # Open and load the image
            image = Image.open(image_path)

            # Preprocess the image  by resizing and normalizing
            image = image.resize((224, 224))
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)

            # Extract the image embedding
            with torch.no_grad():
                image_features = self.model.encode_image(image_input).cpu().detach().numpy()

            list_image_embeddings.append(image_features)

        return list_image_embeddings
    
    def get_text_embedding(self, texts: List[str]) -> ndarray:
        """
        Get the embeddings of a list of texts
        """
        list_text_embeddings = []

        # Tokenize the texts with CLIP 
        text_tokens = clip.tokenize(texts).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens).cpu().detach().numpy()
        
        list_text_embeddings.append(text_features[0])
        return list_text_embeddings
    
