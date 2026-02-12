import streamlit as st
st.write("✅ RAG APP VERSION 2 LOADED")

st.title("Fordham RAG Prototype")
st.write("If you can see this, Streamlit is working.")

question = st.text_input("Ask a question:")

if question:
    st.write("You asked:", question)
