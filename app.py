import streamlit as st
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

Try to answer in markdown format. to format, images or any links as well

Your code will be directly sent to a markdown renderer

Context : {}

Query : {}
        '''

        prompt = prompt.format(context , query)

        messages = [HumanMessage(content = prompt)]

        response = chat.invoke(messages).content

        open('file.txt' , 'w').write(response)

        st.markdown(response)
