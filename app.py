import streamlit as st
from langchain_core.messages import HumanMessage,AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
import base64
import io
from PIL import Image
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os
# import validators
from create_vectorstore import save_vectorDatabase,load_vectorDatabase,process_add_and_pdfs_to_vectorstore
from rag_response import get_rag_response
from load_store import load_vectorDatabase2

load_dotenv()



if __name__ == "__main__":
    # set streamlit UI
    st.set_page_config(page_title="NeuroNote-AI Student Assistant Application",layout="centered")
    st.markdown("📘 Hey there! 👋 I'm NeuroNote AI, your smart study companion.Let's boost your learning — one note at a time!")


    # Memory for chat and vectorstore
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hi! I am your NeuroNote-AI . How can i help you ?")
        ]
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = load_vectorDatabase2()

    
    # in sidebar (upload pdf's and image option)
    with st.sidebar:
        st.title("📓NeuroNote - AI",width="stretch")
        st.subheader("Your personal AI companion for smarter studying.From notes to notifications — everything in one place.")
        uploaded_pdfs = st.file_uploader("Upload one or more PDFs for RAG context", type=["pdf"],accept_multiple_files=True)
        if uploaded_pdfs:
            st.session_state.vectorstore = process_add_and_pdfs_to_vectorstore(uploaded_pdfs,_existing_vectorstore=st.session_state.vectorstore)
            save_vectorDatabase(st.session_state.vectorstore)
            st.success(f"✅ PDF processed and indexed {len(uploaded_pdfs)} PDF(s)")
            

    # display chat history ---
    for msg in st.session_state.chat_history:
        if isinstance(msg,HumanMessage):
            with st.chat_message("user"):
                st.markdown(msg.content)
        else:
            with st.chat_message("assistant"):
                st.markdown(msg.content)



    user_input = st.chat_input("Type your messages....")
    if user_input:
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message('assistant'):
            with st.spinner("Thinking.."):
                try:
                    
                    # if st.session_state.vectorstore and uploaded_pdfs:
                    #     st.write("📄 Based on your uploaded PDF(s):")
                    #     response = get_rag_response(user_input, st.session_state.vectorstore)
                    if st.session_state.vectorstore:
                        st.write("📄 Based on your previously saved documents:")
                        response = get_rag_response(user_input, st.session_state.vectorstore)
                        
                    else:
                        st.write("⚠️ No PDF uploaded or vectorstore not initialized. Please upload PDFs for RAG context.")
                        response = "Please upload PDFs to get a response based on the content."
                    st.markdown(response)
                    st.session_state.chat_history.append(AIMessage(content=response))

                except ValueError:
                    st.error("⚠️ Please provide a valid question or input.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")