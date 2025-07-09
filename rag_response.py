import os 
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain.chains.retrieval_qa.base import RetrievalQA
from dotenv import load_dotenv

load_dotenv()
# from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

RAG_PROMPT_TEMPLATE = """
            You are a helpful and trustworthy AI Medical Assistant. Use the provided medical context to answer the user's question accurately and clearly.

            If the context includes relevant information, use it in your response. If the context does not help answer the question, you may use your own medical knowledge to generate a complete and accurate answer.

            Guidelines:
            - Be concise, clear, and medically accurate.
            - Use simple language unless the question is from a medical professional.
            - If generating an answer beyond the context, do not hallucinate references.
            - Say "Based on your documents..." if the context is used.

            First, answer the user's medical question **using only the information from the provided context**.
            Then, if the context answer is too brief or incomplete, **generate a well-explained version of the answer** using your medical knowledge, ensuring it aligns with the context.
            Use simple, medically accurate language. If a term is complex, explain it in layman's terms.

            ---

            📄 Context:
            {context}

            ❓ User Question:
            {question}

            ---

            ✍️ Your Response:
            1. 📚 **Answer from Documents**:
            (Use only the context above to answer.)

            2. 💡 **Detailed Explanation (Generated)**:
            (Expand and explain based on your knowledge, ensuring clarity and depth.)
            """

def get_rag_response(user_query, vectorstore):
    """
    Get a response from the RAG system using the provided query and vectorstore,
    using custom prompt logic.
    """
    
    if not vectorstore:
        raise ValueError("Vectorstore is not initialized. Please load the vectorstore first.")

    # Set up retriever
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # Initialize Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.4
    )

    # Set up custom prompt
    prompt = PromptTemplate(
        input_variables=["context", user_query],
        template=RAG_PROMPT_TEMPLATE
    )

    # Build RAG chain with custom prompt
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt}
    )

    # Run the chain
    result = rag_chain.invoke({"query": user_query})
    rag_response = result.get("result", "").strip()

    # Check if RAG response is weak (or optionally, check keyword presence or length)
    if not rag_response or "not found" in rag_response.lower() or len(rag_response) < 20:
        ### 2️⃣ Agent Reasoning Fallback
        tools = tools or []
        llm_agent = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.5
        )

        agent_execute = create_react_agent(model=llm_agent, tools=tools)

        response_result = ""
        for chunk in agent_execute.stream({"messages": [HumanMessage(content=user_query)]}):
            if "agent" in chunk and "messages" in chunk["agent"]:
                for message in chunk["agent"]["messages"]:
                    response_result += message.content
        return response_result.strip()
    
    return rag_response
