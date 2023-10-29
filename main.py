import streamlit as st
import os
from langchain_helper import get_qa_chain, create_vector_DB 


st.title('Tech Learning QA 🌳')

if not os.path.isdir('faiss_index'):
    btn = st.button("Create Knowledgebase")
    if btn:
        create_vector_DB()

if os.path.isdir('faiss_index'):
    question = st.text_input('Question: ')

    if question:
        chain = get_qa_chain()
        response = chain(question)
        st.header('Answer: ')
        response['result']

