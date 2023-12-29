import os
import gradio as gr
from embeddings import CLIP
from chromadb import Client, Settings


client = Client(Settings(is_persistent=True, persist_directory="./clip_chroma"))
clip_embedding = CLIP(model_name="ViT-B/32", device="cpu")
coll = client.get_collection(name="clip", embedding_function=clip_embedding.get_image_embedding)

# Retrieve image from query
def retrieve_image_from_query(query: str, image: str) -> str:
    """
    Retrieve an image from a query
    """

    text_embedding = clip_embedding.get_text_embedding(texts=query)
    # Convert the text embeddings to float values
    text_embedding = [float(val) for val in text_embedding]

    # Query the database for using the text embeddings
    results = coll.query(query_embeddings=text_embedding,  include=["documents", "metadatas"], n_results=4)

    # Get the retrieved documents(images) and their metadata
    docs = results["documents"][0]
    description = results["metadatas"][0]

    # Create a list to store the docs and metadata
    list_of_docs = []

    for doc, desc in zip(docs, description):
        list_of_docs.append((doc, list(desc.values())[0]))

    return list_of_docs

def retrieve_image_from_image(image):
    """
    Retrieve an image from an image
    """

    image_name = image.name
    
    # Query the collection using the image file name as the query text
    result = coll.query(
        query_texts=image_name,  # Use the image file name as the query text
        include=["documents", "metadatas"],  # Include both documents and metadata in the results
        n_results=4  # Specify the number of results to retrieve
    )

    docs = result['documents'][0]
    descs = result["metadatas"][0]

    list_of_docs = []

    # Iterate through the retrieved documents and metadata
    for doc, desc in zip(docs, descs):
        # Append a tuple containing the document and its metadata to the list
        list_of_docs.append((doc, list(desc.values())[0]))

    # Return the list of document-metadata pairs
    return list_of_docs

# Function to display an image
def show_img(image):
    return image.name

# Using Gradio blocks, create a UI that allows users to input a query and also upload an image to search for similar images
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            query = gr.Textbox(lines=2, placeholder="Enter a query")
            gr.HTML("OR")
            photo = gr.Image(scale=1)
            button = gr.UploadButton(label="Upload a photo", file_types=["image"])

        with gr.Column():
            gallery = gr.Gallery(scale=1,label="Image Results", show_label=False, object_fit="contain", height="auto", preview=True)

    # Submit query
    query.submit(fn=retrieve_image_from_query, inputs=[query], outputs=[gallery])

    # Upload image
    button.upload(fn=show_img,inputs=[button],outputs=[photo]).then(fn=retrieve_image_from_image, inputs=[button], outputs=[gallery])


if __name__ == '__main__':
    demo.launch(server_name="0.0.0.0", server_port=8080)