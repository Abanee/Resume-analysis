import streamlit as st
import os
from rag_util import load_and_split_documents, build_vector_store, create_qa_chain

st.set_page_config(page_title="Resume Search RAG Bot")
st.title("📄 Resume RAG Chatbot (Codespaces Optimized)")
st.markdown("Upload resume PDFs, and ask about names, skills, or experience.")

UPLOAD_DIR = "resumes"
os.makedirs(UPLOAD_DIR, exist_ok=True)

uploaded_files = st.file_uploader("Upload Resume PDFs", accept_multiple_files=True, type="pdf")

if uploaded_files:
    for file in uploaded_files:
        file_path = os.path.join(UPLOAD_DIR, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
    st.success("✅ Resumes uploaded!")

if st.button("Index Resumes"):
    with st.spinner("🔍 Processing resumes..."):
        docs = load_and_split_documents(UPLOAD_DIR)
        vectordb = build_vector_store(docs)
        qa_chain = create_qa_chain(vectordb)
        st.session_state.qa_chain = qa_chain
        st.success("✅ Resumes indexed! You can now ask questions.")

if "qa_chain" in st.session_state:
    query = st.text_input("🧠 Ask about a candidate or skill:")
    if query:
        with st.spinner("💡 Generating answer..."):
            result = st.session_state.qa_chain(query)
            st.markdown("### 📌 Answer:")
            st.write(result["result"])
            with st.expander("📄 Sources"):
                for src in result["source_documents"]:
                    st.markdown(f"**{src.metadata.get('source', 'Unknown')}**")
                    st.write(src.page_content[:500] + "...")
