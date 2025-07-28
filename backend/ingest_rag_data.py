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
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws", 
            region="us-east-1" 
        )
    )

# Connect to Pinecone index
index = pc.Index(INDEX_NAME)

# Set up LangChain Pinecone vector store with correct embedding model
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = PineconeVectorStore(index=index, embedding=embedding_model, text_key="text")

# Text splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

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
        },
        "2025-07-18": {
            "food": "Breakfast: apple slices with peanut butter. Lunch: spinach and feta wrap. Dinner: baked cod with veggies.",
            "vitals": "Heart rate: 89 bpm. Blood pressure: 118/76. Oxygen saturation: 99%. Step count: 6800.",
            "medical_record": "Stable condition. No new concerns; routine monitoring continues."
        },
        "2025-07-19": {
            "food": "Breakfast: scrambled eggs with spinach. Lunch: chicken salad. Dinner: vegetable curry and rice.",
            "vitals": "Heart rate: 86 bpm. Blood pressure: 117/75. Oxygen saturation: 98%. Step count: 6900.",
            "medical_record": "Mild fatigue reported. Advised extra rest."
        },
        "2025-07-20": {
            "food": "Breakfast: oatmeal with honey and walnuts. Lunch: grilled turkey sandwich. Dinner: salmon and quinoa.",
            "vitals": "Heart rate: 87 bpm. Blood pressure: 116/74. Oxygen saturation: 99%. Step count: 7000.",
            "medical_record": "Energy levels steady. Continue iron supplements."
        },
        "2025-07-21": {
            "food": "Breakfast: fruit salad with yogurt. Lunch: tomato soup with crackers. Dinner: roasted chicken and potatoes.",
            "vitals": "Heart rate: 86 bpm. Blood pressure: 118/76. Oxygen saturation: 98%. Step count: 7200.",
            "medical_record": "No new symptoms. Mild anemia monitored."
        },
        "2025-07-22": {
            "food": "Breakfast: whole wheat toast with jam. Lunch: egg salad wrap. Dinner: grilled shrimp and veggies.",
            "vitals": "Heart rate: 87 bpm. Blood pressure: 119/76. Oxygen saturation: 98%. Step count: 7350.",
            "medical_record": "Reports mild headache in evening. Rest recommended."
        },
        "2025-07-23": {
            "food": "Breakfast: banana pancakes. Lunch: vegetable stir-fry. Dinner: baked tilapia with spinach.",
            "vitals": "Heart rate: 88 bpm. Blood pressure: 117/75. Oxygen saturation: 99%. Step count: 7400.",
            "medical_record": "Headache resolved. No new complaints."
        },
        "2025-07-24": {
            "food": "Breakfast: poached eggs with tomatoes. Lunch: quinoa and black bean salad. Dinner: chicken noodle soup.",
            "vitals": "Heart rate: 86 bpm. Blood pressure: 118/77. Oxygen saturation: 99%. Step count: 7550.",
            "medical_record": "Routine checkup. Mild anemia persists."
        },
        "2025-07-25": {
            "food": "Breakfast: berry smoothie. Lunch: hummus and veggie wrap. Dinner: turkey chili.",
            "vitals": "Heart rate: 87 bpm. Blood pressure: 116/75. Oxygen saturation: 98%. Step count: 7650.",
            "medical_record": "No new issues. Continuing supplements."
        },
        "2025-07-26": {
            "food": "Breakfast: Greek yogurt with honey. Lunch: chicken Caesar salad. Dinner: baked potatoes with veggies.",
            "vitals": "Heart rate: 88 bpm. Blood pressure: 117/74. Oxygen saturation: 99%. Step count: 7700.",
            "medical_record": "Energy good. No symptoms reported."
        },
        "2025-07-27": {
            "food": "Breakfast: avocado toast. Lunch: pasta primavera. Dinner: grilled fish tacos.",
            "vitals": "Heart rate: 89 bpm. Blood pressure: 117/75. Oxygen saturation: 98%. Step count: 7800.",
            "medical_record": "Condition stable. Mild anemia monitored."
        },
        "2025-07-28": {
            "food": "Breakfast: muesli with almond milk.",
            "vitals": "Heart rate: 88 bpm. Blood pressure: 116/74. Oxygen saturation: 99%. Step count: 7900.",
            "medical_record": "Routine monitoring. Patient feels well."
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
        },
        "2025-07-18": {
            "food": "Breakfast: donut and iced coffee. Lunch: bacon cheeseburger with fries. Dinner: skipped.",
            "vitals": "Heart rate: 138 bpm. Blood pressure: 147/96. Oxygen saturation: 90%. Step count: 3100.",
            "medical_record": "Patient reported chest discomfort this morning. Emergency services contacted; awaiting cardiology evaluation."
        },
        "2025-07-19": {
            "food": "Breakfast: bagel with cream cheese. Lunch: pizza slice. Dinner: grilled chicken sandwich.",
            "vitals": "Heart rate: 136 bpm. Blood pressure: 146/95. Oxygen saturation: 91%. Step count: 3000.",
            "medical_record": "Complained of dizziness after breakfast. Cardiology follow-up pending."
        },
        "2025-07-20": {
            "food": "Breakfast: chocolate muffin. Lunch: cheeseburger. Dinner: macaroni and cheese.",
            "vitals": "Heart rate: 138 bpm. Blood pressure: 148/97. Oxygen saturation: 90%. Step count: 3100.",
            "medical_record": "Reported shortness of breath. Emergency consult advised."
        },
        "2025-07-21": {
            "food": "Breakfast: ham and cheese sandwich. Lunch: chicken nuggets. Dinner: steak and mashed potatoes.",
            "vitals": "Heart rate: 137 bpm. Blood pressure: 147/95. Oxygen saturation: 91%. Step count: 3200.",
            "medical_record": "Skipped medication again. Increased monitoring recommended."
        },
        "2025-07-22": {
            "food": "Breakfast: waffles with syrup. Lunch: fried fish sandwich. Dinner: pasta and garlic bread.",
            "vitals": "Heart rate: 140 bpm. Blood pressure: 150/98. Oxygen saturation: 89%. Step count: 2800.",
            "medical_record": "Severe chest pain at noon. 911 called. Hospitalization required."
        },
        "2025-07-23": {
            "food": "Breakfast: fried eggs and toast. Lunch: turkey sandwich. Dinner: chicken noodle soup.",
            "vitals": "Heart rate: 126 bpm. Blood pressure: 138/90. Oxygen saturation: 93%. Step count: 3300.",
            "medical_record": "Returned from hospital. Prescribed new cardiac medication."
        },
        "2025-07-24": {
            "food": "Breakfast: bran cereal. Lunch: BLT sandwich. Dinner: grilled cheese.",
            "vitals": "Heart rate: 128 bpm. Blood pressure: 140/91. Oxygen saturation: 94%. Step count: 3400.",
            "medical_record": "No new symptoms. Monitoring continues."
        },
        "2025-07-25": {
            "food": "Breakfast: cheese omelette. Lunch: ham and turkey club. Dinner: tomato soup and crackers.",
            "vitals": "Heart rate: 122 bpm. Blood pressure: 136/88. Oxygen saturation: 95%. Step count: 3500.",
            "medical_record": "Energy improved. Cardiology follow-up scheduled."
        },
        "2025-07-26": {
            "food": "Breakfast: cinnamon roll. Lunch: chicken salad. Dinner: baked potato with cheese.",
            "vitals": "Heart rate: 125 bpm. Blood pressure: 137/89. Oxygen saturation: 94%. Step count: 3450.",
            "medical_record": "Mild headache reported. Advised to rest."
        },
        "2025-07-27": {
            "food": "Breakfast: bacon and eggs. Lunch: cheeseburger. Dinner: BBQ chicken pizza.",
            "vitals": "Heart rate: 120 bpm. Blood pressure: 135/87. Oxygen saturation: 95%. Step count: 3600.",
            "medical_record": "Condition stabilizing. No chest pain today."
        },
        "2025-07-28": {
            "food": "Breakfast: fruit smoothie.",
            "vitals": "Heart rate: 119 bpm. Blood pressure: 134/86. Oxygen saturation: 96%. Step count: 3700.",
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
