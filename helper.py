import os
import streamlit as st
import pickle
import time
# from langchain import OpenAI
# from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv() # initiallizes all the enviorment variables

def get_result(URLs, main_placeholder, file_path):
    """
    The function `get_result` takes a list of URLs, a placeholder for displaying status, and a file path
    as input. It loads data from the URLs, splits the text into chunks, creates embeddings for the
    chunks using OpenAI, and saves the embeddings to a FAISS index stored in a pickle file.
    
    :param URLs: The URLs parameter is a list of URLs from which you want to load data
    :param main_placeholder: The `main_placeholder` parameter is a placeholder object that is used to
    display status messages or progress updates to the user interface. It could be a text box or any
    other UI element where you want to show the status messages
    :param file_path: The `file_path` parameter is the path where you want to save the FAISS index
    pickle file. It should be a string representing the file path, including the file name and
    extension. For example, if you want to save the file as "index.pickle" in the current directory, you
    can
    """
    
    # show the status
    main_placeholder.text("Data Loading....Started")
    
    # create a loader that just gets all the text data from the URLs...in a document format
    loader = UnstructuredURLLoader(urls=URLs)
    #load data
    data = loader.load()
    
    # showing the status
    main_placeholder.text("Text Splitter....Started")
    # split data into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['/n/n', '/n', '.', ',', ' '],
        chunk_size= 1000
    ) # seperate recursively on the bassis of separaters and maintain max size of 1000 for each chunk
    # split the text
    docs = text_splitter.split_documents(data)
    
    # showing the status
    main_placeholder.text("Embedding....Started")
    # Create embeddings for all those chunks
    embeddings = OpenAIEmbeddings()
    
    # print(f"Embedding size ----> {len(embeddings}")
    # gets the openAi embeddings
    vectorstore_openai = FAISS.from_documents(docs, embeddings)# saving it to FAISS index
    # embeddes my chunks to the embeddings
    
    time.sleep(2)
    
    # save the faiss index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openai, f)
    