import streamlit as st
import PyPDF2
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage
from langchain_cohere import ChatCohere
from video_data import video_data

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

# Function to search for videos based on query
def search_videos(query):
    results = []
    query_lower = query.lower()
    for video in video_data:
        if any(query_lower in tag.lower() for tag in video["tags"]):
            results.append(video["url"])
    return results

# Streamlit app
st.title("Video Search App")

# Input form for user query
query = st.text_input("Enter a query to search for videos:")

# Search and display the videos
if query:
    matched_videos = search_videos(query)
    if matched_videos:
        st.write(f"Found {len(matched_videos)} videos for query: '{query}'")
        for url in matched_videos:
            st.video(url, start_time=0)
    else:
        st.write(f"No videos found for query: '{query}'")
