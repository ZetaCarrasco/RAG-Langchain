from langchain_community.vectorstores import Chroma
from typing import Dict, Any

class Test_Chunks:
    def __init__(self, vector_store: Chroma):
        self.vector_store = vector_store

    def mostrar_primer_chunk_por_documento(self):
        """Muestra el primer chunk de cada documento fuente diferente"""
        try:
            collection = self.vector_store.get()
            if not collection["ids"]:
                print("No hay documentos en el vector store")
                return

            docs_por_fuente: Dict[str, Any] = {}
            for _id, doc_content, metadata in zip(
                collection["ids"],
                collection["documents"],
                collection["metadatas"]
            ):
                fuente = metadata.get("source", "desconocido")
                if fuente not in docs_por_fuente:
                    docs_por_fuente[fuente] = {
                        "id": _id,
                        "content": doc_content,
                        "metadata": metadata
                    }

            print("\n--- PRIMER CHUNK DE CADA DOCUMENTO ---")
            for i, (fuente, datos) in enumerate(docs_por_fuente.items(), 1):
                print(f"\nDocumento {i}: {fuente}")
                print(f"ID: {datos['id']}")
                print("Contenido:")
                print(datos["content"][:500] + "...")
                print("\n---")
            print(f"\nTotal de documentos procesados: {len(docs_por_fuente)}")
        except Exception as e:
            print(f"Error al recuperar chunks: {e}")
