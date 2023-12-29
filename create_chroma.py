import os
import sys
from embeddings import CLIP
from chromadb import Client, Settings


if __name__ == '__main__':
    # Load captions into a dictionary
    captions_dict = {}
    with open("dataset/archive/reduced_captions.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            image_path, caption = line.strip().split(',', 1)
            captions_dict[image_path] = caption

    # Iterate over images
    image_paths = []
    image_ids = []
    descriptions = []
    for img_path in os.listdir("dataset/archive/Images"):
        image_paths.append(os.path.join("dataset/archive/Images", img_path))
        image_id = img_path.split(".")[0]
        image_ids.append(image_id)

        # Look up caption in dictionary
        if img_path in captions_dict:
            descriptions.append({"image_path": img_path, "captions": captions_dict[img_path]})

    
    clip_embedding = CLIP(model_name="ViT-B/32", device="cpu")
    client = Client(settings = Settings(is_persistent=True, persist_directory="./clip_chroma"))
    coll = client.get_or_create_collection(name = "clip", embedding_function = clip_embedding.get_image_embedding)
    coll.add(ids=image_ids,
             documents = image_paths,
             metadatas = descriptions)
    
    print('Done with loading the database')
    sys.exit(0)