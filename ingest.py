
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from src.supabase_client import supabase

# Step 1: Load PDF
loader = PyPDFLoader("data/company_policy.pdf")
documents = loader.load()

print("PDF loaded:", len(documents), "pages")

# Step 2: Split PDF into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print("Total chunks:", len(chunks))

# Step 3: Load embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Step 4: Generate embeddings and insert into Supabase
records = []

for chunk in chunks:
    vector = embeddings.embed_query(chunk.page_content)

    records.append({
        "content": chunk.page_content,
        "metadata": chunk.metadata,
        "embedding": vector
    })

# Step 5: Insert records in batches
batch_size = 50

for start in range(0, len(records), batch_size):
    batch = records[start:start + batch_size]

    supabase.table("documents").insert(batch).execute()

print("Successfully stored", len(records), "chunks in Supabase!")

