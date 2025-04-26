import os
from typing import List
from langchain.schema import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

class Vector_Store:
    def __init__(self, embedding_model_base: str = "llama3"):
        # OllamaEmbeddings against your local Ollama server
        self.embedding_model = OllamaEmbeddings(
            base_url="http://localhost:11434",
            model=embedding_model_base
        )

    def create_vector_store(
        self,
        documents: List[Document],
        persist_directory: str = None
    ) -> Chroma:
        if not documents:
            print("Nenhum documento fornecido para criar o vector store.")
            return None

        if persist_directory:
            print("Criando novo Vector Store...")
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding= self.embedding_model,
                persist_directory=persist_directory
            )
            vector_store.persist()
            print(f"Vector store persistido em: {persist_directory}")
            return vector_store

        # in‐memory fallback
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model
        )
        print("Vector store criado na memória.")
        return vector_store

    def load_vector_store(self, persist_directory: str) -> Chroma:
        if (
            persist_directory
            and os.path.isdir(persist_directory)
            and os.listdir(persist_directory)
        ):
            print(f"Carregando vector store existente de: {persist_directory}")
            return Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embedding_model
            )
        print(f"Diretório '{persist_directory}' não encontrado ou vazio. Nenhum vector store carregado.")
        return None
