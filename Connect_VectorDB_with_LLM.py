from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

# Setup LLM (Mistral with HuggingFace)
HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(hf_repo_id):
    llm_model = HuggingFaceEndpoint(
        repo_id = hf_repo_id,
        temperature = 0.5,
        huggingfacehub_api_token = HF_TOKEN,
        max_new_tokens = 512
    )
    return llm_model

# custome prompt template for llm 
CUSTOM_PROMPT_TEMPLATE = """
Answer the question strictly based on the provided context.  
Do not include any information that is not present in the context.  
If the answer is not in the context, respond with "I don't know."

Context: {context}  
Question: {question}  

Provide only the answer.

"""

# set custom prompot template in PromptTemplate
def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables = ["context","question"])
    return prompt


# load vectoreDB 
DB_FAISS_PATH = "vectorembedding/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model,allow_dangerous_deserialization = True)

# create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm = load_llm(HUGGINGFACE_REPO_ID),
    chain_type = "stuff",
    retriever = db.as_retriever(search_kwargs = {"k":3}),
    return_source_documents = True,
    chain_type_kwargs = {'prompt' : set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)


# how invoke with a single query
user_query =  input("Write Query Here: ")
response = qa_chain.invoke({'query':user_query})
print("Result : ", response['result'])
print("Source Documents: ", response['source_documents'])