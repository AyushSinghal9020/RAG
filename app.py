import streamlit as st
from PIL import Image
import requests
from io import BytesIO
from image_data import image_data
import PyPDF2
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage
from langchain_cohere import ChatCohere

file = st.file_uploader('Upload a PDF file' , type = ['pdf'])

if file is not None : 

    pdf = PyPDF2.PdfReader(file)

    text = ' '.join([
        pdf.pages[page_number].extract_text()
        for page_number 
        in range(len(pdf.pages))
    ])

    chunks = [
        text[index : index + 1024]
        for index 
        in range(0 , len(text) , 1024)
    ]

    chat = ChatCohere(cohere_api_key = 'FELFXgLGfcqsy4eh4Q75dXNT7VyIQjKZmhkiIug3')
    embeddings = HuggingFaceEmbeddings(model_name = 'all-MiniLM-L6-v2')
    vectorstore = FAISS.from_texts(chunks , embeddings)

    query = st.text_input('Enter your query')

    if st.button('Search') :

        similar_docs = vectorstore.similarity_search(query)
        context = ' '.join([
            similar_doc.page_content
            for similar_doc
            in similar_docs
        ])

        prompt = '''
You are a conversational chatbot, your task is to answer questions based on the context provided.

Context : {}

Query : {}
        '''

        prompt = prompt.format(context , query)

        messages = [HumanMessage(content = prompt)]
        
        response = chat.invoke(messages).content

        st.write(response)

# Function to load image from a URL
def load_image(url):
    response = requests.get(url)
    response.raise_for_status()
    img = Image.open(BytesIO(response.content))
    return img

# Function to search for images based on query
def search_images(query):
    results = []
    query_lower = query.lower()
    for image in image_data:
        if any(query_lower in tag.lower() for tag in image["tags"]):
            results.append(image["url"])
    return results

# Streamlit app
st.title("Image Search App")

# Input form for user query
query = st.text_input("Enter a query to search for images:")

# Search and display the images
if query:
    matched_images = search_images(query)
    if matched_images:
        st.write(f"Found {len(matched_images)} images for query: '{query}'")
        for url in matched_images:
            img = load_image(url)
            st.image(img, caption=url.split('/')[-1], use_column_width=True)
    else:
        st.write(f"No images found for query: '{query}'")
