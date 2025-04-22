from langchain_community.vectorstores import Chroma
from typing import Dict, Any
from langchain_core.documents import Document

class Test_Chunks:
    def __init__(self, vector_store:Chroma):
        self.vector_store = vector_store

    def mostrar_primer_chunk_por_documento(self):
        """Muestra el primer chunk de cada documento fuente diferente"""
        try:
            # Obtenemos todos los documentos del almacén vectorial
            collection = self.vector_store.get()
            
            if not collection['ids']:
                print("No hay documentos en el vector store")
                return
            
            # Diccionario para agrupar por fuente (nombre de archivo)
            documentos_por_fuente: Dict[str, Dict[str, Any]] = {}
            
            for id, doc_content, metadata in zip(collection['ids'], 
                                                collection['documents'], 
                                                collection['metadatas']):
                fuente = metadata.get('source', 'desconocido')
                
                if fuente not in documentos_por_fuente:
                    documentos_por_fuente[fuente] = {
                        'id': id,
                        'content': doc_content,
                        'metadata': metadata
                    }
            
            # Mostramos el primer chunk de cada fuente
            print("\n--- PRIMER CHUNK DE CADA DOCUMENTO ---")
            for i, (fuente, datos) in enumerate(documentos_por_fuente.items(), 1):
                print(f"\nDocumento {i}: {fuente}")
                print(f"ID: {datos['id']}")
                print("Contenido:")
                print(datos['content'][:500] + "...")  # Muestra solo los primeros 500 caracteres
                print("\n---")
                
            print(f"\nTotal de documentos procesados: {len(documentos_por_fuente)}")
            
        except Exception as e:
            print(f"Error al recuperar chunks: {e}")

        
            
       

    def buscar_chunks(self, consulta: str, k: int = 5):
        print(f"\nProcurando chunks para: '{consulta}' (top {k})")
        try:
               documentos = self.vector_store.similarity_search(consulta, k=k)
               if not documentos:
                    print("Nehum chunk encontrado")
                    return
               for i, doc in enumerate(documentos):
                    print(f"\n Chunk {i+1}     ")
                    print(doc.page_content.strip())
                    print(f"Fonte: {doc.metadata.get('source', 'desconhecida')}")
        except Exception as e:
             print(f"Erro ao buscar chunks: {e}")            
        