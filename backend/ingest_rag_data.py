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

# Sample data (multi-day, per-user, per-category)
daily_sample_data = {
    "user_mary": {
        "2025-07-14": {
            "food": "Breakfast: oatmeal and berries. Lunch: quinoa salad. Dinner: grilled salmon and steamed veggies.",
            "vitals": "Heart rate: 85 bpm. Blood pressure: 117/75. Oxygen saturation: 99%. Step count: 6200.",
            "medical_record": "Mild anemia. Iron supplement ongoing."
        },
        "2025-07-15": {
            "food": "Breakfast: yogurt with granola. Lunch: lentil soup. Dinner: stir-fried tofu and rice.",
            "vitals": "Heart rate: 88 bpm. Blood pressure: 118/76. Oxygen saturation: 98%. Step count: 6400.",
            "medical_record": "Stable condition. Energy levels improving."
        },
        "2025-07-16": {
            "food": "Breakfast: banana smoothie. Lunch: vegetable wrap. Dinner: chicken stew.",
            "vitals": "Heart rate: 86 bpm. Blood pressure: 116/74. Oxygen saturation: 98%. Step count: 6500.",
            "medical_record": "Reports slight fatigue. Advised hydration."
        },
        "2025-07-17": {
            "food": "Breakfast: toast and almond butter. Lunch: chickpea salad. Dinner: miso soup.",
            "vitals": "Heart rate: 87 bpm. Blood pressure: 117/75. Oxygen saturation: 99%. Step count: 6700.",
            "medical_record": "Symptoms mild. No concerns noted."
        }
    },
    "user_john": {
        "2025-07-14": {
            "food": "Breakfast: sausage and biscuits. Lunch: pepperoni pizza. Dinner: burger and soda.",
            "vitals": "Heart rate: 125 bpm. Blood pressure: 140/89. Oxygen saturation: 94%. Step count: 3000.",
            "medical_record": "History of arrhythmia. Missed medication dose."
        },
        "2025-07-15": {
            "food": "Breakfast: pancakes with syrup. Lunch: fried chicken. Dinner: skipped. Low water intake.",
            "vitals": "Heart rate: 132 bpm. Blood pressure: 142/91. Oxygen saturation: 92%. Step count: 2800.",
            "medical_record": "Reported chest discomfort at 3 PM. Risk flagged."
        },
        "2025-07-16": {
            "food": "Breakfast: bacon and toast. Lunch: hotdog. Dinner: spaghetti and cola.",
            "vitals": "Heart rate: 118 bpm. Blood pressure: 139/88. Oxygen saturation: 94%. Step count: 3500.",
            "medical_record": "Advised urgent monitoring. Skipped check-in."
        },
        "2025-07-17": {
            "food": "Breakfast: croissant and coffee. Lunch: fast food. Dinner: skipped again.",
            "vitals": "Heart rate: 135 bpm. Blood pressure: 145/93. Oxygen saturation: 91%. Step count: 2900.",
            "medical_record": "Condition deteriorating. Emergency care advised."
        }
    }
}

# Ingest into Pinecone
for user_id, day_data in daily_sample_data.items():
    for day, categories in day_data.items():
        for data_type, content in categories.items():
            chunks = splitter.split_text(content)
            for i, chunk in enumerate(chunks):
                metadata = {
                    "user_id": user_id,
                    "data_type": data_type,
                    "date": day  # used for filtering in RAG
                }
                vectorstore.add_texts([chunk], metadatas=[metadata], ids=[f"{user_id}_{data_type}_{day}_{i}"])

print("✅ Multi-day ingestion complete.")
