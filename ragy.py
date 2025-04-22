from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


class RAGY:
    def __init__(self, vector_store: Chroma, llm_model: str = "llama3:latest"):
        try:
            self.vector_store = vector_store
            self.llm = OllamaLLM(base_url="http://localhost:11434", model= llm_model)
           
           #Prompt em português para que as respostan que gere sejam em portugês
            prompt_template = PromptTemplate.from_template(
               """Você é um assistente que responde perguntas com base no contexto abaixo.
               Responda sempre em portugês, ,es,o qie o contexto esteja em outo idioma.
               
               Contexto.
               {context}

               Pergunta:
               {question}


               Resposta:"""
           )
            self.rag_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(search_kwargs={"k": 10}),
                chain_type_kwargs={"prompt": prompt_template},
                return_source_documents=True,
            )
            print("RAG inicializado com sucesso!")
        except Exception as e:
            print(f"Erro ao inicializar o RAG: {e}")
            self.vector_store = None
            self.llm = None
            self.rag_chain = None
            

    def query(self, question: str) ->str:
        if self.rag_chain:
            try:
                result = self.rag_chain.invoke({"question": question})
                return result["result"]
            except Exception as e:
                return f"Erro ao consultar RAG: {e}"
            else:
                return "O RAG não foi inicializado corretamente."       

              
