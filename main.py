import os
from documents_process import DocumentsProcess
from vector_store import Vector_Store
from test_chunks import Test_Chunks
from ragy import RAGY

def main():
    input_folder = "./selectedDocs"
    persist_dir = "./chroma_db"

    #Inicializa o gerenciador de vetores
    vectorstore_creator = Vector_Store(embedding_model_base="llama3")
    vector_store = vectorstore_creator.load_vector_store(persist_dir)
    
    #Cria o vector store a partir dos documentos em casso de não houver
    if vector_store is None:
        # Se não existir vector store, carregue os documentos e crie um novo
        pdf_processor = DocumentsProcess(chunk_size=500, chunk_overlap=100)
        documents = pdf_processor.load_and_process(input_folder)
                   
        if not documents:
            print("Nenhum documento processado. Encerrando.")
            return

        vector_store = vectorstore_creator.create_vector_store(documents, persist_directory=persist_dir)

        if not vector_store:
            print("Falha ao criar o vector store. Encerrando.")
            return
    else:
        print("Vector store carregado com sucesso!")


    
    inspector = Test_Chunks(vector_store)
    inspector.mostrar_primer_chunk_por_documento()

    # Inicializar o  RAG (agora ele usa o vector_store carregado ou criado)
    rag_pipeline = RAGY(vector_store=vector_store, llm_model="llama3")

    if not rag_pipeline.rag_chain:
        print("Falha ao inicializar o pipeline RAG. Encerrando.")
        return

    while True:
        pergunta = input("Digite sua pergunta (ou 'sair' para encerrar): ")
        if pergunta.lower() == 'sair':
            break
        
        #Mostra os chunks dos documentos recuperados
        print("\n Documentos recuperados:")
        docs = vector_store.similarity_search(pergunta, k=3)
        for i, doc in enumerate(docs):
            print(f"\n --- Documento {i+1} ---")
            print(doc.page_content[:800])
        

        resposta = rag_pipeline.query(pergunta)
        print(f"Pergunta: {pergunta}")
        print(f"Resposta: {resposta}\n")

if __name__ == "__main__":
    main()