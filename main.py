import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from chromadb.config import Settings
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
import torch
from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails import LLMRails
from rails_wrapper import GeminiLangchainLLM


# === Load environment variables ===
load_dotenv()
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")

# === Initialize LLM ===
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    max_tokens=30000
)
config = RailsConfig.from_path("rails")

# Wrap Gemini with callable class
gemini_wrapped_llm = GeminiLangchainLLM(llm)

# Initialize Guardrails
guardrails = LLMRails(config, llm=gemini_wrapped_llm)
guardrails.llm = gemini_wrapped_llm
# === Embeddings ===
embeddings = HuggingFaceEmbeddings(model_name=r"C:\models\bll-MiniLM-L6-v2")

# === Load and Split PDF ===
pdf_path = "C:\\Users\\MX513TW\\Final_Proj\\FY24_Q1_Consolidated_Financial_Statements.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
splits = text_splitter.split_documents(documents)

# === Vectorstore ===
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    client_settings=Settings(anonymized_telemetry=False, persist_directory="./chroma_db"),
    collection_name="rag_collection"
)
retriever = vectorstore.as_retriever()

# === Prompts for contextualization and QA ===
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given a chat history and the latest user question which might reference context in the chat history, "
               "formulate a standalone question. Don't answer it, just rephrase if necessary."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an assistant for question-answering. Use the following context to answer concisely:\n\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# === Message history store ===
store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)

# === Multi-query Expansion Prompt ===
query_expansion_prompt = PromptTemplate(
    input_variables=["query"],
    template="""Generate 5 semantically diverse and relevant expanded versions of the following question.
Only output the list of queries (no explanation), one per line:\n\nQuery: {query}\n\nExpanded Queries:"""
)

def expand_queries(query, llm):
    output = llm.invoke(query_expansion_prompt.format(query=query))
    #output = llm.invoke(query_expansion_prompt.format(query=query))
    queries = [line.strip() for line in output.content.split("\n") if line.strip()]

    #queries = [line.strip() for line in output.split("\n") if line.strip()]
    return queries

# === Query Re-Ranker ===
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device='cuda' if torch.cuda.is_available() else 'cpu')

def rerank_queries(original_query, expanded_queries):
    pairs = [(original_query, q) for q in expanded_queries]
    scores = reranker.predict(pairs)
    best_query = expanded_queries[scores.argmax()]
    return best_query, list(zip(expanded_queries, scores))

# === Streamlit UI ===
st.set_page_config(page_title="Gemini RAG QA Chat", layout="wide")
st.title("📄💬 Financial PDF Chatbot")

if "session_id" not in st.session_state:
    st.session_state.session_id = "default_session"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Ask a question from the PDF...")

if user_input:
    session_id = st.session_state.session_id
    session_history = get_session_history(session_id)

    # Step 1: Expand user query to multiple candidates
    expanded_queries = expand_queries(user_input, llm)
    st.markdown("🔍 **Expanded Queries (before re-ranking):**")
    for q in expanded_queries:
        st.markdown(f"- {q}")

    # Step 2: Re-rank queries based on relevance to original input
    best_query, scored_queries = rerank_queries(user_input, expanded_queries)
    st.success(f"🏆 **Selected Query for Retrieval:** {best_query}")

    with st.expander("📊 Query Scores"):
        for q, s in scored_queries:
            st.markdown(f"**Score:** {s:.4f} → {q}")

    # Step 3: Use best query in conversational chain
    response = conversational_rag_chain.invoke(
        {"input": best_query},
        config={"configurable": {"session_id": session_id}}
    )
    answer = response["answer"]

    # Step 4: Update chat history
    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("ai", answer))

# === Render chat history ===
for sender, msg in st.session_state.chat_history:
    with st.chat_message(sender):
        st.markdown(msg)


# rag_response = conversational_rag_chain.invoke(
#     {"input": best_query},
#     config={"configurable": {"session_id": session_id}}
# )
# raw_answer = rag_response["answer"]

# # Pass the raw answer through Guardrails for validation/modification
# guardrails_response = guardrails.generate(prompt=raw_answer)
# answer = guardrails_response["content"]
