# import streamlit as st
# from langchain_core.messages import HumanMessage,AIMessage
# from langchain.tools import tool
# from langchain.chains import RetrievalQA
# from dotenv import load_dotenv
# import os
# from create_vectorstore import save_vectorDatabase,load_vectorDatabase,process_add_and_pdfs_to_vectorstore
# from rag_response import get_rag_response
# from load_store import load_vectorDatabase2
# from generate_text import generate_text_response
# from create_vectorstore import process_add_and_pdfs_to_vectorstore, load_vectorDatabase

# load_dotenv()

# # Inject CSS styling for chat bubbles and sidebar
# # --- Custom CSS Injection ---
# # def inject_css():
# #     st.markdown(
# #         """
# #         <style>
# #         /* Custom CSS from style.css will be loaded here */
# #         </style>
# #         """,
# #         unsafe_allow_html=True
# #     )
# #     # Load content of style.css and inject it
# #     with open("style.css") as f:
# #         st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# if __name__ == "__main__":

#     # inject_css() # Inject CSS at the very beginning
#     # set streamlit UI
#     st.set_page_config(page_title="HelixMedica – Intelligent Medical Support", layout="centered")
#     st.markdown("🧠 **HelixMedica**: Bridging Science & Care 🤖\n\nHi! I'm your AI assistant trained on medical knowledge and trusted documents. Ask anything — from symptoms to science.")


#     # Memory for chat and vectorstore
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = [
#             AIMessage(content="Hi! I am HelixMedica, your AI Medical Assistant. How can I help you today?")
#         ]
    
#     if "vectorstore" not in st.session_state:
#         st.session_state.vectorstore = None

    
#     # in sidebar (upload pdf's and image option)
#     with st.sidebar:
#         st.title("🧬 HelixMedica – AI", anchor="top")
#         st.subheader("Your intelligent AI-powered medical assistant.\n\nFrom symptoms to science — always here to help you understand your health better.")

#         uploaded_pdfs = st.file_uploader("Upload one or more PDFs for RAG context", type=["pdf"],accept_multiple_files=True)
#         if uploaded_pdfs:
#             st.session_state.vectorstore = process_add_and_pdfs_to_vectorstore(uploaded_pdfs,_existing_vectorstore=st.session_state.vectorstore)
#             save_vectorDatabase(st.session_state.vectorstore)
#             st.success(f"✅ PDF processed and indexed {len(uploaded_pdfs)} PDF(s)")
            

#     # display chat history ---
#     for msg in st.session_state.chat_history:
#         if isinstance(msg,HumanMessage):
#             with st.chat_message("user"):
#                 st.markdown(msg.content)
#         else:
#             with st.chat_message("assistant"):
#                 st.markdown(msg.content)



#     user_input = st.chat_input("Type your messages....")
#     if user_input:
#         st.session_state.chat_history.append(HumanMessage(content=user_input))
#         with st.chat_message("user"):
#             st.markdown(user_input)

#         with st.chat_message('assistant'):
#             with st.spinner("Thinking.."):
#                 try:
                    
#                     if uploaded_pdfs:
#                         if st.session_state.vectorstore:
#                             st.write("📄 Based on your uploaded PDF(s):")
#                             response = get_rag_response(user_input, st.session_state.vectorstore1)
#                         else:
#                             st.session_state.vectorstore = load_vectorDatabase()
                        
#                     else:
#                         # st.write("📄 Based on your previously saved documents:")
#                         if st.session_state.vectorstore:
#                             st.session_state.vectorstore = load_vectorDatabase2()
#                             rag_data = get_rag_response(user_input, st.session_state.vectorstore) if st.session_state.vectorstore else None
                            
#                             response = generate_text_response(  
#                                 user_input,
#                                 st.session_state.chat_history,
#                                 "Medical Assistant",
#                                 rag_data
#                             )
#                         else:
#                             rag_data = get_rag_response(user_input, st.session_state.vectorstore)
#                             response = generate_text_response(
#                                 user_input,
#                                 st.session_state.chat_history,
#                                 "Medical Assistant",
#                                 rag_data
#                             )
                    
#                     st.markdown(response)
#                     st.session_state.chat_history.append(AIMessage(content=response))

#                 except ValueError:
#                     st.error("⚠️ Please provide a valid question or input.")
#                 except Exception as e:
#                     st.error(f"❌ Error: {str(e)}")




import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools import tool
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os
from create_vectorstore import (
    save_vectorDatabase,
    load_vectorDatabase,
    process_add_and_pdfs_to_vectorstore
)
from rag_response import get_rag_response
from load_store import load_vectorDatabase2
from generate_text import generate_text_response

load_dotenv()

st.set_page_config(
    page_title="HelixMedica – Intelligent Medical Support",
    layout="centered"
)

st.markdown("""
🧠 **HelixMedica**: Bridging Science & Care 🤖

Hi! I'm your AI assistant trained on medical knowledge and trusted documents. Ask anything — from symptoms to science.
""")

# Session state initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hi! I am HelixMedica, your AI Medical Assistant. How can I help you today?")
    ]

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# Sidebar for uploading PDFs
with st.sidebar:
    st.title("🧬 HelixMedica – AI", anchor="top")
    st.subheader("Your intelligent AI-powered medical assistant.\n\nFrom symptoms to science — always here to help you understand your health better.")

    uploaded_pdfs = st.file_uploader("Upload one or more PDFs for RAG context", type=["pdf"], accept_multiple_files=True)
    if uploaded_pdfs:
        st.session_state.vectorstore = process_add_and_pdfs_to_vectorstore(
            uploaded_pdfs,
            _existing_vectorstore=st.session_state.vectorstore
        )
        save_vectorDatabase(st.session_state.vectorstore)
        st.success(f"✅ Processed and indexed {len(uploaded_pdfs)} PDF(s)")

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant"):
        st.markdown(msg.content)

# Chat input handling
user_input = st.chat_input("Type your messages....")
if user_input:
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = ""
                if uploaded_pdfs and st.session_state.vectorstore:
                    st.write("📄 Based on your uploaded PDF(s):")
                    response = get_rag_response(user_input, st.session_state.vectorstore)

                else:
                    if not st.session_state.vectorstore:
                        st.session_state.vectorstore = load_vectorDatabase2()

                    rag_data = get_rag_response(user_input, st.session_state.vectorstore)
                    response = generate_text_response(
                        user_input,
                        st.session_state.chat_history,
                        "Medical Assistant",
                        rag_data
                    )

                st.markdown(response)
                st.session_state.chat_history.append(AIMessage(content=response))

            except ValueError:
                st.error("⚠️ Please provide a valid question or input.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
