import PyPDF2

import streamlit as st

import google.generativeai as genai

from gradio_client import Client , file
from pdf2image import convert_from_path
from langchain.schema.document import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage
from langchain_cohere import ChatCohere

file = st.file_uploader('Upload file' , type = ['pdf'])
query = st.text_input('Search Query')


if file : 

    documents = []

    open('file.pdf', 'wb').write(file.getbuffer())   

    images = convert_from_path('file.pdf')

    genai.configure(api_key = 'AIzaSyCQdd-2-hoFLiSxIZaIYQkCyipRR4_z51g')

    model = genai.GenerativeModel('gemini-pro-vision')


    with st.spinner('Generating Image Captions ') : 


        for image in images :         

            image.save(f'image.png' , 'PNG')
            try : 

                response = model.generate_content(
                    [
                        '' , 
                        image
                    ])
            
                response = response.text

            except : response = 'Couldnt generate caption for this image'

            documents.append(Document(
                page_content = response , 
                metadata = {
                    'type' : 'image' , 
                    'image' : image
                }
            ))

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

    for chunk in chunks : 

        documents.append(
            Document(
                page_content = chunk , 
                metadata = {'type' : 'text'}
            ))

    with st.spinner('Creating Vector Store') :

        embeddings = HuggingFaceEmbeddings(model_name = 'all-MiniLM-L6-v2')

        vc = FAISS.from_documents(
            documents ,  
            embedding = embeddings
        )

    if query : 

        with st.spinner('Searching ') : 

            similar_docs = vc.similarity_search(query , 5)

            context = ''
            for doc in similar_docs : 

                if doc.metadata['type'] == 'image' : 

                    st.image(doc.metadata['image'])
                    
                    context += doc.page_content

                else : context += doc.page_content

            prompt = '''
You are a conversational chatbot, your task is to answer questions based on the context provided.

Context : {}

Query : {}
            '''

            chat = ChatCohere(cohere_api_key = 'FELFXgLGfcqsy4eh4Q75dXNT7VyIQjKZmhkiIug3')
            prompt = prompt.format(context , query)

            messages = [HumanMessage(content = prompt)]
            
            response = chat.invoke(messages).content

            st.write(response)
