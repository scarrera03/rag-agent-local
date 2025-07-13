from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# === Configuration ===
CHROMA_DIR = "db"
DATA_DIR = "data"

# === 1. Load documents ===
loader = DirectoryLoader(DATA_DIR, glob="**/*.txt")
documents = loader.load()

# === 2. Chunking ===
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# === 3. Embeddings ===
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# === 4. Persistent vector store ===
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=CHROMA_DIR
)
retriever = vectorstore.as_retriever()

# === 5. Connect to local LLM (LM Studio must be running) ===
llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",  # LM Studio ignores this, but LangChain requires it
    model="mistral-7b-instruct-v0.1.Q4_0",
    temperature=0.7
)

# === 6. Create QA chain (RAG) ===
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# === 7. Simple console interface ===
print("\n🤖 RAG Agent ready. Type a question or 'exit' to quit.\n")

while True:
    query = input("🧠 Question: ")
    if query.lower() in ["exit", "quit"]:
        break
    result = qa_chain(query)
    print(f"\n✅ Answer:\n{result['result']}\n")
