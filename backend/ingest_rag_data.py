# ingest_rag_data.py

import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables from .env file
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")  # e.g., "gcp-starter"
INDEX_NAME = "elderly-health-agent"

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if it doesn't exist (512 for text-embedding-3-small)
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=512,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",  # or "gcp" if that's your region
            region="us-east-1"  # or your Pinecone region
        )
    )

# Connect to Pinecone index
index = pc.Index(INDEX_NAME)

# Set up LangChain Pinecone vector store with correct embedding model
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = PineconeVectorStore(index=index, embedding=embedding_model, text_key="text")

# Text splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

# Sample data
sample_data = {
    "user_mary": {
        "food": "Breakfast: oatmeal with banana and honey. Lunch: grilled chicken salad with vinaigrette. Dinner: lentil soup, whole grain bread, and herbal tea.",
        "vitals": "Heart rate: 88 bpm. Blood pressure: 118/76. Oxygen saturation: 98%. Step count: 6200 steps today.",
        "medical_record": "Last visit: May 10, 2024. Diagnosis: Mild anemia. Prescribed: Iron supplements for 3 months. No history of chronic disease."
    },
    "user_john": {
        "food": "Breakfast: bacon, eggs, and white bread toast. Lunch: cheeseburger with fries and soda. Dinner: skipped. Water intake: low.",
        "vitals": "Heart rate peaked at 132 bpm around 3:15 PM. Blood pressure: 142/91. Oxygen saturation dropped to 92%. Step count: 4300 steps.",
        "medical_record": "Last visit: April 25, 2024. Diagnosis: Cardiac arrhythmia. Previous ER visit due to chest pain. Prescribed: Beta-blockers."
    }
}

# Ingest into Pinecone
for user_id, categories in sample_data.items():
    for data_type, content in categories.items():
        chunks = splitter.split_text(content)
        for i, chunk in enumerate(chunks):
            metadata = {
                "user_id": user_id,
                "data_type": data_type,
                "timestamp": "2024-07-13T10:00:00"
            }
            vectorstore.add_texts([chunk], metadatas=[metadata], ids=[f"{user_id}_{data_type}_{i}"])

print("✅ Ingestion complete. Pinecone is ready.")
