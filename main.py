import os
import sys
import shutil
from documents_process import DocumentsProcess
from vector_store     import Vector_Store
from test_chunks      import Test_Chunks
from ragy             import RAGY

def main(force_rebuild: bool = False):
    input_folder = "./selectedDocs"
    persist_dir  = "./chroma_db"

    # ▶️ force‐rebuild if requested
    if force_rebuild and os.path.isdir(persist_dir):
        shutil.rmtree(persist_dir)
        print(f"🔥 Cleared persist dir: {persist_dir}")

    vs_creator   = Vector_Store(embedding_model_base="llama3")
    vector_store = vs_creator.load_vector_store(persist_dir)

    # 1) If store missing, build it from scratch
    if vector_store is None:
        print("🆕 Building new vector store…")
        pdf_proc = DocumentsProcess(chunk_size=1000, chunk_overlap=200)
        docs     = pdf_proc.load_and_process(input_folder)
        if not docs:
            print("Nenhum documento processado. Encerrando.")
            return
        vector_store = vs_creator.create_vector_store(
            documents=docs,
            persist_directory=persist_dir,
        )

    # 2) Else — load existing and only index _new_ PDFs
    else:
        print("🔄 Loaded existing vector store. Checking for new PDFs…")
        collection       = vector_store.get()
        existing_sources = {m.get("source") for m in collection["metadatas"]}

        all_pdfs = [
            f for f in os.listdir(input_folder)
            if f.lower().endswith(".pdf")
        ]
        to_index = [f for f in all_pdfs if f not in existing_sources]

        if to_index:
            pdf_proc   = DocumentsProcess(chunk_size=1000, chunk_overlap=200)
            new_chunks = []
            for fname in to_index:
                print(f"Processing new file: {fname}")
                fullpath = os.path.join(input_folder, fname)
                text     = pdf_proc._extract_text_from_pdf(fullpath)
                if not text:
                    continue
                chunks = pdf_proc._chunk_text(text, fname)
                new_chunks.extend(chunks)

            if new_chunks:
                vector_store.add_documents(new_chunks)
                vector_store.persist()
                print(f"✅ Indexed {len(new_chunks)} chunks from {len(to_index)} new PDFs.")
        else:
            print("👍 No new PDFs to index.")

    # Sanity: show first chunk per source
    inspector = Test_Chunks(vector_store)
    inspector.mostrar_primer_chunk_por_documento()

    # Build RAG pipeline
    rag = RAGY(vector_store=vector_store, llm_model="llama3:latest")
    if not rag.rag_chain:
        print("Falha ao inicializar RAG. Encerrando.")
        return

    # Interactive ask loop
    while True:
        pergunta = input("Pergunta (ou 'sair'): ")
        if pergunta.lower() == "sair":
            break

        docs = vector_store.similarity_search(pergunta, k=3)
        for i, d in enumerate(docs, start=1):
            print(f"\n— Doc {i} ({d.metadata['source']}):")
            print(d.page_content[:500], "…")

        resposta = rag.query(pergunta)
        print(f"\n🤖 Resposta: {resposta}\n")


if __name__ == "__main__":
    fr = "--force-rebuild" in sys.argv
    main(force_rebuild=fr)
