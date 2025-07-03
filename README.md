# 📄💬 Financial PDF Chatbot (RAG-Powered with Query Expansion + Re-Ranking)

An interactive **Streamlit-based chatbot** for querying information from a financial PDF using a **Retrieval-Augmented Generation (RAG)** pipeline. It supports **query expansion**, **semantic re-ranking**, and uses **LangChain**, **Chroma**, **Guardrails**, and **Gemini 2.5** for safe, contextual responses.


---

## 📁 Project Structure

```bash
.
├── main.py               # Main Streamlit app with RAG pipeline
├──venv                   # virtual environment
├── rails_wrapper.py        # Custom Gemini LLM wrapper for NeMo Guardrails
├── chroma_db/              # Persisted Chroma vectorstore
├── FY24_Q1_Consolidated...# PDF to be queried
├── rails/
│   ├── colang/main.cl      # NeMo Guardrails logic
│   ├── config.yml          # Guardrails config
│   └── rails.json          # Rails metadata
├── .env                    # Environment variables
├── requirements.txt        # Python dependencies
└── README.md               # Documentation
```

# 🚀 Features
#### 🔍 Retrieval-Augmented Generation (RAG) for PDF-based QA

#### 🧠 Multi-query expansion to improve retrieval recall

#### 🏆 Cross-encoder re-ranking for selecting the most relevant query

#### 📚 Contextual chat history memory across turns

#### 🛡️ Integrated NeMo Guardrails for safety and compliance

#### ⚡ Powered by Google Gemini 2.5 Flash

# 🛠️ Tech Stack


| Layer        | Tools / Frameworks                                           |
|--------------|---------------------------------------------------------------|
| Frontend     | Streamlit                                                    |
| Retrieval    | LangChain Retriever, Chroma Vectorstore, HuggingFace Embeddings |
| Reranking    | SentenceTransformers CrossEncoder (`ms-marco-MiniLM`)        |
| Guardrails   | NVIDIA NeMo Guardrails                                       |
| LLM Provider | Google Gemini 2.5 Flash (via LangChain)                      |
| Runtime      | Python 3.10+, venv, python-dotenv                            |


# ⚙️ Setup Instructions
## 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-repo/gemini-rag-chatbot.git
cd gemini-rag-chatbot
```

## 2️⃣ Create and Activate Virtual Environment
```bash
python -m venv venv
source venv/bin/activate          # On Windows: venv\Scripts\activate
```

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```
Tip: If you encounter CUDA issues with SentenceTransformers, ensure your PyTorch installation matches your GPU/CPU environment.

# 🔑 Environment Variables
Create a .env file in the project root:

```bash
GOOGLE_API_KEY=your_google_api_key
HF_TOKEN=your_huggingface_token
```
# 🗃️ PDF Data Loading
By default, the app loads this PDF:
```bash
FY24_Q1_Consolidated_Financial_Statements.pdf
```
You can replace pdf_path in main.py to point to any other document.

# 💡 How It Works
### 1)User Question

You enter a natural language query about the PDF.

### 2)Query Expansion

The model generates 5 semantically diverse reformulations.

### 3)Reranking

A cross-encoder scores them against the original query, selecting the most relevant one.

### 4)Retrieval

Chroma vectorstore retrieves context passages.

### 5)Answer Generation

Gemini generates a concise answer using the retrieved context.

### 6)Guardrails

NeMo validates and post-processes the response.

# ▶️ Running the Application
## Start Streamlit:

```bash
streamlit run main.py
#It will launch on http://localhost:8501.
```

# 🧪 Live Example:
## User:

```yaml
What is the total net sales?
```
## Expanded Queries:
<img width="589" alt="image" src="https://github.com/user-attachments/assets/463c4b7f-1f65-4440-b0a5-6ba724f3bb67" />

## Selected Query for Retrieval:

<img width="589" alt="image" src="https://github.com/user-attachments/assets/cbfd6321-0a2d-4641-a4fd-8f0ce4047365" />

## Query scores with cross encoder

<img width="370" alt="image" src="https://github.com/user-attachments/assets/6e514d14-2fb6-4f32-b85c-04983fb2f537" />

## Answer:

<img width="837" alt="image" src="https://github.com/user-attachments/assets/4bc348b3-1961-4ec4-aee4-0b88e913aa46" />


# 🧼 Debugging Tips
 #### Confirm .env has valid API keys.
 #### Make sure chroma_db/ directory is writable.
 #### If CUDA is unavailable, CrossEncoder will fall back to CPU automatically.
 #### Errors are printed in the Streamlit console.



