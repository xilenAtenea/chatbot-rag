import os
import tempfile
import dotenv  # type: ignore
import streamlit as st  # type: ignore

from rag_logic import load_pdf, doc_embeddings, retrieve_chunks, model_response


# configuración Langsmith
dotenv.load_dotenv()
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING")
os.environ["USER_AGENT"] = os.getenv("USER_AGENT")

st.set_page_config(page_title="Chatbot RAG", layout="centered")
st.title("Chatbot RAG")
st.write("¡Hazle preguntas directas a tus propios PDFs! \nEste chatbot te permite cargar documentos y obtener respuestas instantáneas.")

if "document_indexado" not in st.session_state:
    st.session_state["document_indexado"] = False
    st.session_state["chat_history"] = []
    st.session_state["vector_store"] = None
    st.session_state["filename"] = None

# Paso 1: subir PDF
st.header("1️. Sube un documento PDF")
uploaded_file = st.file_uploader("Selecciona un PDF", type=["pdf"])

if uploaded_file:
    st.session_state["filename"] = uploaded_file.name
    st.success(f"Archivo cargado: {uploaded_file.name}")

# Paso 2: parámetros del modelo y explicación para usuario
st.header("2️. Ajusta el comportamiento del modelo")
st.markdown("""
- **Temperatura:** controla qué tan creativas o seguras son las respuestas.  
  → Baja = más directas y confiables. Alta = más imaginativas y variadas.

- **Top-k:** limita cuántas opciones puede considerar el modelo al escribir cada palabra.  
  → Más bajo = respuestas más predecibles. Más alto = más variedad.

- **Top-p:** elige entre solo las palabras más probables, hasta alcanzar cierto porcentaje de confianza.  
  → Más bajo = respuestas más controladas. Más alto = más libertad al responder.
""")

with st.form("model_config"):
    col1, col2, col3 = st.columns(3)
    with col1:
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05)
    with col2:
        top_p = st.slider("Top-p", 0.0, 1.0, 0.8, 0.05)
    with col3:
        top_k = st.slider("Top-k", 1, 100, 40, 1)
    submitted = st.form_submit_button("Procesar documento")

if submitted and uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_pdf_path = tmp_file.name

    try:
        with st.spinner("Procesando el documento..."):
            splits = load_pdf(temp_pdf_path)
            st.session_state["vector_store"] = doc_embeddings(splits)
            st.session_state["document_indexado"] = True
        st.success("Documento indexado con éxito.")
    except Exception as e:
        st.error(f"Error al procesar: {str(e)}")
    finally:
        os.remove(temp_pdf_path)

# Paso 3: chat activado
if st.session_state["document_indexado"]:
    st.header(f"3️. ¡Pregunta sobre: {st.session_state['filename']}!")

    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Escribe tu pregunta...")

    if user_input:
        st.session_state["chat_history"].append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.status("Buscando contexto...", expanded=True) as status:
            retrieved_docs, n, raw_chunks = retrieve_chunks(
                st.session_state["vector_store"], user_input
            )
            status.write(f"{n} fragmentos recuperados.")

        with st.chat_message("assistant"):
            with st.spinner("Generando respuesta..."):
                try:
                    stream, meta = model_response(
                        retrieved_docs, user_input,
                        top_p=top_p,
                        temperature=temperature,
                        top_k=top_k
                    )
                    response = st.write_stream(stream)
                except Exception as e:
                    response = f"Error: {str(e)}"
                    st.error(response)

        with st.expander(">Parámetros del modelo"):
            st.markdown(f"""
            - `temperature`: `{meta['temperature']}`
            - `top_p`: `{meta['top_p']}`
            - `top_k`: `{meta['top_k']}`
            """)
        with st.expander(">Fragmentos utilizados"):
            for i, (doc, score) in enumerate(raw_chunks):
                st.markdown(f"**Chunk {i+1} (score: {round(score, 3)}):**")
                st.code(doc.page_content)

        st.session_state["chat_history"].append({"role": "assistant", "content": response})
