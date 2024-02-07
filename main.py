
import os

from dotenv import load_dotenv
load_dotenv()

# print(GOOGLE_API_KEY)
import urllib
import warnings
from pathlib import Path as p
from pprint import pprint
from IPython.display import display
from IPython.display import Markdown
import textwrap
import pandas as pd
import pickle
import chromadb
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st


def get_data_from_pdf():
    pdf_file = "intel_book.pdf"
    pdf_loader = PyPDFLoader(pdf_file)
    pages = pdf_loader.load_and_split()
    # print(context)
    return pages

def text_to_chunks(pages):
    text_spllitter = RecursiveCharacterTextSplitter(chunk_size = 1200, chunk_overlap = 10)
    # context = "\n\n".join(str(p.page_content) for p in pages)
    text_chunks = text_spllitter.split_documents(pages)
    # print(text_chunks)
    return text_chunks

def get_context_data(text_chunks, query):
    # Check for existing directory and load vector index if found
    persist_directory = "db"
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    if os.path.exists(persist_directory):
        vectordb = Chroma(persist_directory="db", embedding_function=embedding)
    
    else:
        vectordb = Chroma.from_documents(documents=text_chunks, embedding=embedding,
                                        persist_directory=persist_directory)
        print("Created new vector index and persisted to directory.")

    # Persisting already handled within `from_documents` and `as_retriever`
    retriever = vectordb.as_retriever()

    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # vector_index = Chroma.from_texts(text_chunks, embeddings).as_retriever()
    # retrieve_docs = vector_index.get_relevant_documents(query)
    # # print(retrieve_docs)
    return retriever


def get_conv_chain(retriever):
    
    prompt_template = """You are a helpful and informative bot that answers questions using text from the reference passage included below.
    Be sure to respond in a complete sentence, being comprehensive, including all relevant background information.
    However, you are talking to a non-technical audience, so be sure to break down complicated concepts and
    strike a friendly and converstional tone. You can answer from other sources but try to focus on given context. If the passage is completely irrelevant to the answer, you may ignore it and say "I can't find answer in given PDF".
    PASSAGE: '{context}'
    QUESTION: '{question}'       
    ANSWER:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5, convert_system_message_to_human=True)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context","question"])
    
    qa_chain = RetrievalQA.from_chain_type(llm = model,
                                       chain_type = "stuff",
                                       retriever = retriever,
                                       return_source_documents = True)
    
    
    
    # chain = load_qa_chain(model, chain_type = "stuff", prompt = prompt)
    return qa_chain


def process_llm_responce(model_output):
    print(model_output['result'])
    print('\n\n Sources : ')
    for source in model_output["source_documents"]:
        print(source.metadata)
        print(source) #need to edit based on what to retrieve. change the return function  to retrived metadata.
    return model_output['result']



def user_input(text):
    data = get_data_from_pdf()
    chunks = text_to_chunks(data)
    retriever = get_context_data(chunks, text)
    chain = get_conv_chain(retriever)
    # responce = chain({"input_documents": context_doc, "question": text}, return_only_outputs=True)
    responce = chain(text)
    processed_ouptut = process_llm_responce(responce)
    return processed_ouptut


def main():
    
    st.title("Vishay")

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"roles":"Assistant", "content":"Hey Buddy, how may I help you? Shoot your questions here"}]


    for message in st.session_state.messages:
        with st.chat_message(message['roles']):
            st.write(message['content'])

    if prompt := st.chat_input(placeholder="Type your questions before"):
        st.session_state.messages.append({'roles':'user', 'content':prompt})
        with st.chat_message("user"):
            st.write(prompt)
    
    if st.session_state.messages[-1]["roles"] != "Assistant":
        with st.chat_message("Assistant"):
            with st.spinner("Thinking . . . . . ."):
                output = user_input(prompt)
                st.write(output)
                
                
        message = {"roles":"Assitant", "content": output}
        st.session_state.messages.append(message)
    
if __name__ == "__main__":
    main()