import os
from langchain_community.document_loaders import PyPDFLoader #type:ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter #type:ignore
from langchain_ollama import OllamaEmbeddings, ChatOllama #type:ignore
from langchain_chroma import Chroma #type:ignore
from langchain_core.messages import HumanMessage #type:ignore


def load_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        add_start_index=True
    )
    return splitter.split_documents(docs)

def doc_embeddings(splits, persist_dir: str = "../chroma_db", collection_name: str = "collection"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    persist_path = os.path.join(base_dir, persist_dir)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_path
    )

    existing_ids = vector_store.get()["ids"]
    if existing_ids:
        vector_store.delete(ids=existing_ids)
        print("Vector store limpio antes de reindexar.")

    vector_store.add_documents(splits)
    print(f"{len(splits)} chunks indexados.")

    return vector_store

def retrieve_chunks(vector_store, query: str, k: int = 4):
    results = vector_store.similarity_search_with_score(query, k=k)
    retrieved_docs = "\n\n".join([doc.page_content for doc, _ in results])
    return retrieved_docs, len(results), results

def model_response(retrieved_docs: str, query: str, top_p=0.8, temperature=0.1, top_k=40):
    llm = ChatOllama(
        model="mistral",
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=200
    )

    messages = [
        HumanMessage(
            content=f"""Responde únicamente con base en el siguiente contexto.
                Si no puedes responder con certeza, responde:
                "No tengo suficiente información en el documento para responder con certeza."

                Contexto:
                {retrieved_docs}

                Pregunta:
                {query}
                """
                        )
                    ]

    stream = llm.stream(messages)

    return stream, {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k
    }
