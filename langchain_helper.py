from langchain.llms import GooglePalm
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

import google.generativeai as palm

from dotenv import load_dotenv
import os
load_dotenv()


embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
llm = GooglePalm()


vector_db_path = 'faiss_index'

def get_qa_chain():

    db = FAISS.load_local(vector_db_path,embeddings)
    retriever = db.as_retriever()

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type = 'stuff',
        input_key="query",
        chain_type_kwargs={'prompt':PROMPT}
    )

    return chain


def create_vector_DB():
    loader = CSVLoader(file_path="faqs.csv", source_column="prompt")
    data = loader.load()
    db = FAISS.from_documents(data, embeddings)
    db.save_local(vector_db_path)



