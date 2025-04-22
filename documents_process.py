import os
from langchain.text_splitter import RecursiveCharacterTextSplitter # type: ignore
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from PyPDF2 import PdfReader
from typing import List


class DocumentsProcess:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_and_process(self, folder_path: str) -> List[Document]:
        all_documents = []
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"A pasta '{folder_path}' não existe.")
        
        pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('pdf')]
        if not pdf_files:
            print(f"Aviso: Nenhum arquivo PDF encontrado em '{folder_path}'")
            return all_documents

        for file_name in pdf_files:
            if file_name.endswith('.pdf'):
                print(f"Processando: {file_name}")
                file_path = os.path.join(folder_path, file_name)
            try:
                text = self._extract_text_from_pdf(file_path)
                if text:
                    chunks = self._chunk_text(text, file_name)
                    all_documents.extend(chunks)
                    print(f"→ {len(chunks)} chunks criados")
                else:
                    print(f"Aviso: Nenhum texto extraído de '{file_name}'")
            except Exception as e:
                print(f"Erro ao processar '{file_name}': {str(e)}")
                continue
                    
        return all_documents
    

    def _extract_text_from_pdf(self, file_path: str) -> str:
        text = ""
        try:
            with open(file_path, 'rb') as file:
                reader = PdfReader(file)
                if len(reader.pages) ==0:
                    print(f"Aviso: PDF vazio em '{file_path}'")
                    return ""
                
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page.extract_text() + "\n"

        except Exception as e:
            print(f"Erro ao ler PDF '{file_path}': {e}")
        return text

    def _chunk_text(self, text: str, source: str) -> List[Document]:
        chunks = []
        start = 0
        text_length = len(text)
        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            chunk = text[start:end].strip()
            if chunk:
               metadata = {"source": source}
               chunks.append(Document(page_content=chunk, metadata=metadata))

            start += self.chunk_size - self.chunk_overlap

            if end == text_length:
                break
        print(f"Numero de chunks: {len(chunks)}")    
        return chunks
    

dp = DocumentsProcess()
docs = dp.load_and_process("./selectedDocs")   
print(len(docs)) 
    
"""
processor = DocumentsProcess(chunk_size=500, chunk_overlap=100)
documents = processor.load_and_process("./selectedDocs")
print(f"Número de documentos/chunks gerados: {len(documents)}")
if documents:
    print(f"Primeiro chunk:\n{documents[0].page_content[:100]}...")
    print(f"Metadata do primeiro chunk: {documents[0].metadata}")

"""
    



        
