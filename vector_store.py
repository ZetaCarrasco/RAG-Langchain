from typing import List
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os


class Vector_Store:
    def __init__(self, embedding_model_base: str = "llama3"):
        self.embedding_model = OllamaEmbeddings(base_url="http://localhost:11434", model=embedding_model_base)

    def create_vector_store(self, documents: List[Document], persist_directory: str = None):
       if not documents:
           print("Nenhum documento fornecido para criar o vetor store.")
           return None
       
       if persist_directory:
           if os.path.exists(persist_directory) and os.listdir(persist_directory):
               print(f"Carregando vector store existente de: {persist_directory}")
               vector_store = Chroma(persist_directory=persist_directory, embedding_function=self.embedding_model)
               print("Adicionando novos documentos ao vector store existente...")
               vector_store.add_documents(documents)

           else:
               print("Criando novo Vector Store...")
               vector_store= Chroma.from_documents(documents, self.embedding_model, persist_directory=persist_directory)
               
               vector_store.persist()
               print(f"Vector store persistido em: {persist_directory}")
           return vector_store
       
       else:
           vector_store = Chroma.from_documents(documents, self.embedding_model)
           print("Vector store criado na memória.")
           return vector_store

    #Carrega um vector store persistido.
    def load_vector_store(self, persist_directory: str):
        if persist_directory and os.path.exists(persist_directory) and os.listdir(persist_directory):
            print(f"Carregando vector store existente de: {persist_directory}")
            return Chroma(persist_directory=persist_directory, embedding_function=self.embedding_model)
        else:
            print(f"Diretório '{persist_directory}' não encontrado ou vazio. Nenhum vector store carregado.")
            return None
