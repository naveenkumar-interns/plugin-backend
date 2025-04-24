## added follow up question and increased speed of response
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import ast
from datetime import datetime
import pymongo
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from typing import List
from pymongo import MongoClient
from multiprocessing import Pool, cpu_count
from pymongo.operations import UpdateOne
import tqdm
import time
from pinecone import Pinecone
from pinecone import ServerlessSpec  # Import ServerlessSpec
import numpy as np  # Import NumPy for handling NaN


hf_token = "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"
MONGO_URI = "mongodb+srv://jsckson_store:jsckson_store@cluster0.9a981.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(MONGO_URI)

model = HuggingFaceInferenceAPIEmbeddings(api_key=hf_token, model_name="sentence-transformers/all-MiniLM-L6-v2")

Pinecone_index = None

app = Flask(__name__)
CORS(app)

def build_embedding(database_name,collection_name):
    hf_token = "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"
    MONGO_URI = "mongodb+srv://jsckson_store:jsckson_store@cluster0.9a981.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    client = MongoClient(MONGO_URI)
    db = client[database_name]
    collection = db[collection_name]


    # Step 3: Load the embedding model
    model = HuggingFaceInferenceAPIEmbeddings(api_key=hf_token, model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Process documents in chunks
    chunk_size = 1000
    total_docs = collection.count_documents({})
    embeddings = []

    for skip in range(0, total_docs, chunk_size):
        documents = list(collection.find().skip(skip).limit(chunk_size))
        print(f"Processing documents {skip + 1} to {skip + len(documents)}")
        
        def generate_embedding_batch(doc_batch):
            try:
                texts = []
                valid_docs = []
                for doc in doc_batch:
                    desc = doc.get('expanded_description', '')
                    if desc and isinstance(desc, str):
                        texts.append(desc)
                        valid_docs.append(doc)
                    else:
                        print(f"Skipping document {doc['_id']}: Invalid or missing expanded_description")
                if not texts:
                    return []
                time.sleep(0.5)  # Avoid HuggingFace rate limits
                embeddings = model.embed_documents(texts)
                return [(doc['_id'], embedding) for doc, embedding in zip(valid_docs, embeddings)]
            except Exception as e:
                print(f"Error processing batch: {e}")
                return []
        
        def parallel_embed(documents, batch_size=32, num_processes=1):  # Single process for free tiers
            batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]
            print(f"Processing {len(batches)} batches with {num_processes} processes")
            with Pool(processes=num_processes) as pool:
                results = list(tqdm.tqdm(pool.imap(generate_embedding_batch, batches), total=len(batches)))
            return [item for sublist in results for item in sublist]
        
        batch_embeddings = parallel_embed(documents, batch_size=32, num_processes=1)
        embeddings.extend(batch_embeddings)

    # Bulk update
    bulk_chunk_size = 1000
    bulk_operations = [
        UpdateOne({'_id': doc_id}, {'$set': {'embedding': embedding}})
        for doc_id, embedding in embeddings
    ]

    print(f"Performing bulk write for {len(bulk_operations)} documents")
    for i in range(0, len(bulk_operations), bulk_chunk_size):
        try:
            collection.bulk_write(
                bulk_operations[i:i + bulk_chunk_size],
                ordered=False
            )
            print(f"Updated {min(i + bulk_chunk_size, len(bulk_operations))} documents")
        except Exception as e:
            print(f"Error during bulk write: {e}")

def build_pinecone_vectorstore(database_name, collection_name):
    MONGO_URI = "mongodb+srv://jsckson_store:jsckson_store@cluster0.9a981.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    Pinecone_api_key = "pcsk_3XafLm_SDzfsZm5fmrpXPafmpaUaJydXGr4KucreMZpGay5Uz84MAY4mk9tKYqKTeNUMrp"
    client = MongoClient(MONGO_URI)
    db = client[database_name]
    collection = db[collection_name]
    # Step 2: Retrieve existing documents
    documents = list(collection.find())  # Adjust limit as needed

    # Step 4: Initialize Pinecone client with your API key
    pc = Pinecone(api_key=Pinecone_api_key)

    # Step 5: Delete existing index if it exists
    index_name = collection_name
    if index_name in pc.list_indexes().names():
        pc.delete_index(index_name)

    # Step 6: Recreate index
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",  # Use cosine similarity (common for embeddings)
        spec=ServerlessSpec(cloud='aws', region='us-east-1')  # Use ServerlessSpec for serverless deployment
    )
    index = pc.Index(index_name)

    # Step 7: Fetch data from MongoDB in batches and upsert to Pinecone
    batch_size = 500
    documents = collection.find().batch_size(batch_size)
    to_upsert = []

    for doc in documents:
        # Use _id as the unique identifier since 'id' may not exist
        id_value = f"id_{str(doc['_id'])}"
        embedding = doc['embedding']  # Use 'embedding' to match earlier code
        # Clean metadata: Remove unwanted fields and handle NaN/None
        metadata = {k: v for k, v in doc.items() if k not in ['_id', 'embedding']}
        # Replace NaN/None with compatible values
        metadata = {
            k: "" if v is None or (isinstance(v, float) and np.isnan(v)) else v
            for k, v in metadata.items()
        }
        # Convert all values to strings
        metadata = {k: str(v) for k, v in metadata.items()}
        
        to_upsert.append({"id": id_value, "values": embedding, "metadata": metadata})
        
        # Upsert in batches
        if len(to_upsert) >= batch_size:
            index.upsert(vectors=to_upsert)
            to_upsert = []

    # Step 8: Upsert any remaining documents
    if to_upsert:
        index.upsert(vectors=to_upsert)

    global Pinecone_index
    Pinecone_index = index

    print("Pinecone index reset and data upserted successfully!")


# Function to get recommendations
def get_recommendations(query, top_k=30):
    # Generate query embedding (384 dimensions)
    query_embedding = model.embed_query(query)
    result = Pinecone_index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return [match['metadata'] for match in result['matches']]


@app.route('/build_embeddings', methods=['POST'])
def get_database_and_collection():
    data = request.json
    database_name = data.get('database_name')
    collection_name = data.get('collection_name')
    if not database_name or not collection_name:
        return jsonify({"error": "Please provide both database_name and collection_name"}), 400
    try:
        build_embedding(database_name, collection_name)
    except Exception as e:
        return jsonify({"error": str(e)+"error in build_embedding"	}), 500
    
    try:
        build_pinecone_vectorstore(database_name, collection_name)
    except Exception as e:
        return jsonify({"error": str(e)+"error in build_pinecone_vectorstore"}), 500
    
    return jsonify({"message": "Embeddings built and stored in Pinecone successfully!"}), 200
    

@app.route('/test', methods=['POST'])
def test():
    if Pinecone_index is None:
        return jsonify({"error": "Pinecone index not initialized. Please run /build_embeddings first."}), 400
    data = request.json
    query = data.get('query')
    if not query:
        return jsonify({"error": "Please provide a query"}), 400
    try:
        recommendations = get_recommendations(query)
        return jsonify({"recommendations": recommendations}), 200
    except Exception as e:
        return jsonify({"error": f"Error in get_recommendations: {str(e)}"}), 500
    


GOOGLE_API_KEY = "AIzaSyDR5hSTYjo6jbiTpHw8AEKZsuRVEEFcAJk"
hf_token = "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"
MONGO_URI = "mongodb+srv://jsckson_store:jsckson_store@cluster0.9a981.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

model_1 = "gemini-1.5-flash"
model_2 = "gemini-2.0-pro-exp-02-05"
model_3 = "gemini-2.0-flash-lite"


client = pymongo.MongoClient(MONGO_URI)
db = client["jacksonHardwareDB"]
collection = db["inventory"]
user_client = pymongo.MongoClient("mongodb+srv://sudhakaran:URvEVWjORGTkaeaq@cluster0.znyhl.mongodb.net/chatbot?retryWrites=true&w=majority&appName=Cluster0")
user_db = user_client["chatbot"]
chat_history_collection = user_db["chats"]



app = Flask(__name__)
CORS(app)


llm = ChatGoogleGenerativeAI(
    model=model_1,
    temperature=0.7,
    max_tokens=60,
    timeout=None,
    max_retries=2,
    google_api_key=GOOGLE_API_KEY,
)



embedding_cache ={}

def generate_embedding(text: str) -> List[float]:
    if text in embedding_cache:
        return embedding_cache[text]
    embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=hf_token, model_name="sentence-transformers/all-MiniLM-L6-v2")
    response = embeddings.embed_query(text)
    embedding_cache[text]=response
    return response


def convert_to_json(data):
    result = []
    forai = []
    for product in data:
        # Filter out unnecessary keys from metadata
        product_info = {
        'id': product.get('id'),
        'title': product.get('title'),
        'description': product.get('description'),
        'product_type': product.get('product_type'),
        'link': product.get('link'),
        'image_list': product.get('image_list'),
        'price': product.get('price'),
        'inventory_quantity': product.get('inventory_quantity'),
        'vendor': product.get('vendor')
        }
        result.append(product_info)

    # print(result)

    return result,forai


def get_product_search(query, top_k=30):
    query_embedding = model.embed_query(query)
    result = Pinecone_index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    results = [match['metadata'] for match in result['matches']]
    return convert_to_json(results)

def analyze_intent(query,store_name,store_description):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite",  # Assuming this is a valid model; adjust if needed
        temperature=0.7,
        max_tokens=60,
        timeout=None,
        max_retries=2,
        google_api_key="AIzaSyDR5hSTYjo6jbiTpHw8AEKZsuRVEEFcAJk"
    )
    try:
        prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a senior data analyst for a {store_name} chatbot. Analyze the user query and categorize the intent as:
        1. **'product'** (asking about products, product availability, product details, or scenario-based queries related to using or purchasing products),
        2. **'website/company'** (asking about the website, company, store operations, services, policies, or FAQs),
        3. **'general'** (queries unrelated to products, website, or company).

        Return only the category name (e.g., "product", "website/company", or "general"). Do not include preambles, explanations, or additional text.

        **Examples:**

        - Query: "I need a hammer for hanging a shelf." → product
        - Query: "What dresses do you have in size medium?" → product
        - Query: "Do you have organic apples?" → product
        - Query: "I’m looking for running shoes under $50." → product
        - Query: "I need ingredients for a vegan cake." → product
        - Query: "How do I fix a leaky faucet?" → product
        - Query: "I’m building a treehouse. What tools do I need?" → product
        - Query: "What outfits are good for a summer wedding?" → product
        - Query: "My car broke down. Do you sell car jacks?" → product
        - Query: "Any deals on laptops?" → product
        - Query: "Where can I find promotions going on?" → website/company
        - Query: "Can I visit your store to see the products?" → website/company
        - Query: "How does your website work?" → website/company
        - Query: "What’s your return policy?" → website/company
        - Query: "Do you deliver?" → website/company
        - Query: "What are your store hours?" → website/company
        - Query: "Can I rent equipment from your store?" → website/company
        - Query: "What payment methods do you accept?" → website/company
        - Query: "Can I special order an item?" → website/company
        - Query: "Do you offer gift cards?" → website/company
        - Query: "How do I apply for a job at your store?" → website/company
        - Query: "Why is the sky blue?" → general
        - Query: "What’s the history of your company?" → website/company
        - Query: "Why do we connect?" → general

        **FAQs to Categorize as 'website/company':**
        - Queries about store locations, hours, or showrooms
        - Queries about delivery, returns, or payment methods
        - Queries about promotions, discounts, or loyalty programs
        - Queries about job applications or company information
        - Queries about website functionality or online ordering
        - Queries about services like repairs, rentals, or special orders
        """
    ),
    ("human", "{query}")
])

        chain = prompt | llm
        response = chain.invoke({"query": query, "store_name": store_name})	
        return response.content.strip()
    except Exception as e:
        print(f"Error in analyze_intent: {str(e)}")
        raise

def research_intent(chat_history,store_name,store_description):
    llm = ChatGoogleGenerativeAI(
    model=model_1,
    temperature=0.7,
    max_tokens=60,
    timeout=None,
    max_retries=2,
    google_api_key=GOOGLE_API_KEY
    )

    try:
        prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a senior research assistant for a {store_name} chatbot. Analyze chat history to track the user's current topic (e.g., clothing, footwear, groceries, hardware). Accumulate filters (e.g., product type, size, color, price) until the topic shifts to a new project or product category, then reset filters. Respond with a single phrase (max 10 tokens) summarizing the latest request, prioritizing the most recent input.

        **Examples:**
        1. History:
           User: 'I need to fix a leaky faucet. What tools?' Bot: 'Pipe wrench or faucet kit?' User: 'Tools to disassemble.' Bot: 'Hand or power tools?' User: 'Hand tools, simple.'
           Output: 'simple hand tools for faucet'
        2. History:
           User: 'I’m buying a dress for a party.' Bot: 'What style dress?' User: 'Evening gown, size medium.' Bot: 'What color?' User: 'Red, under $100.'
           Output: 'red medium evening gown'
        3. History:
           User: 'I need running shoes.' Bot: 'Men’s or women’s shoes?' User: 'Men’s, size 10.' Bot: 'Any brand preference?' User: 'Nike, black.'
           Output: 'black Nike men’s running shoes'
        4. History:
           User: 'I need ingredients for a vegan cake.' Bot: 'Flour or sugar?' User: 'Vegan flour, gluten-free.' Bot: 'What quantity?' User: '5 pounds.'
           Output: 'gluten-free vegan flour'
        5. History:
           User: 'I want to hang a shelf.' Bot: 'Screws or wall anchors?' User: 'Wall anchors for drywall.' Bot: 'Weight capacity?' User: 'Heavy, 50 pounds.' Bot: 'Need a drill?' User: 'Yes, cordless.'
           Output: 'cordless drill for heavy shelf'
        6. History:
           User: 'I need a heater.' Bot: 'Gas or electric?' User: 'Electric, 220V.' Bot: 'Size preference?' User: 'Small, black.' Bot: 'Check heater stock?' User: 'Show me tables.'
           Output: 'tables'
        7. History:
           User: 'Show me lighting options.' Bot: 'Ceiling or outdoor?' User: 'Ceiling, 500 lumens.' Bot: 'Style preference?' User: 'Show me storage cabinets.'
           Output: 'storage cabinets'

        Analyze the conversation and return the summarizing phrase."""
    ),
    ("human", "{chat_history}")
])



        chain = prompt | llm

        response = chain.invoke({"chat_history": chat_history,"store_name":store_name})


        print("\n",response.content)

        return response.content.strip()
    except Exception as e:
        print(f"Error in research_intent: {str(e)}")
        raise

def prioritize_products(user_intent,products):
    products = list(products)    

    print("\n\nproducts : ", products)

    

    llm = ChatGoogleGenerativeAI(
    model=model_3,
    temperature=0.7,
    max_tokens=50000,
    timeout=None,
    max_retries=2,
    google_api_key=GOOGLE_API_KEY,
)

    input_str = f"User asks for : '{user_intent}'\n Products we have: {json.dumps(products, indent=2)}"
    try:
        prompt = """ Role: You are a Product Prioritization Expert that returns ONLY: id. specializing in ranking products based on user intent, price constraints, and relevance.
            
            Your task is to filter, reorder, and return the most relevant products that match the user's intent and budget, colour , product type and other features.

            Rules for Prioritization:
            1. **Match User Intent**: 
            - Prioritize products that contain keywords from the user's intent in the title, description, or product type.
            - Stronger keyword matches (e.g., exact matches in the title) should rank higher.

            2. **Apply Price Constraints**:
            - If a price limit is specified (e.g., "under $30"), exclude products exceeding this threshold.
            - If no price limit is provided, ignore this rule.

            3. **Sort Order**:
            - First, sort by **intent relevance** (strongest keyword matches first).
            - Then, sort by **price** (low to high) within products of equal relevance.

            4. **Output Format**:
            - Return a JSON array of the relevant products, do not alter input data values, only filter and reorder, place all unrelated items from the list.
            -Return ONLY id field in JSON:


        Examples:
        Example 1
        Intent: 'touchscreen gloves'
        Products:
        [
        {
            "id": 4,
            "title": "Touchscreen Gloves",
            "price": "29.99",
            "inventory_quantity": 2,
            "description": "Touchscreen",
            "product_type": "Gloves",
            "link": "https://jacksonshardware.com/touchscreen-gloves",
            "image_list": ["https://jacksonshardware.com/touch1.jpg", "https://jacksonshardware.com/touch2.jpg"],
            "vendor": "TechWear"
        },
        {
            "id": 5,
            "title": "Work Gloves",
            "price": "15.00",
            "inventory_quantity": 4,
            "link": "https://jacksonshardware.com/work-gloves",
            "description": "Rugged"
        }
        ]
        Output:
        [
        {
            "id": 4,
        },
        {
            "id": 5,
        }
        ]

        Example 2
        Intent: 'yeti products'
        Products:
        [
        {
            "id": 8602398097560,
            "title": "YETI® Rambler® Magslider™ Lid",
            "price": "10.00",
            "inventory_quantity": 1,
            "description": "Dishwasher safe",
            "link": "https://jacksonshardware.com/yeti-lid",
            "image_list": ["https://jacksonshardware.com/yeti1.jpg"],
            "vendor": "YETI®"
        },
        {
            "id": 8602860781720,
            "title": "YETI® 14 oz Mug",
            "price": "30.00",
            "inventory_quantity": 0,
            "link": "https://jacksonshardware.com/yeti-mug",
            "image_list": ["https://jacksonshardware.com/mug1.jpg", "https://jacksonshardware.com/mug2.jpg"]
        }
        ]
        Output:
        [
        {
            "id": 8602398097560,
        },
        {
            "id": 8602860781720,
        }
        ]

        Task Execution:
        Now, apply these rules to the following product dataset and return the top most relevant products in sorted JSON format:

        """ + input_str



        # Format the input string correctly and pass it as the 'input' variable
  
        response = llm.invoke(prompt)
        prompt = ""
  
        # print("AI product result :",response.content.replace("\n", "").replace("```json", "").replace("```", "").replace("'", '"').strip())

        id_list = str(json.loads(response.content.replace("\n", "").replace("```json", "").replace("```", "").replace("'", '"').strip()))

        id_list = ast.literal_eval(id_list)

        id_list = [i.get('id') for i in id_list]
        products = sum(products, [])

           
        result = []
        for i in products:
            if i.get("id") in id_list:
                result.append(i)


        return result
    
    except Exception as e:
        print(f"Error in prioritize_products: {str(e)}")
        raise


def get_response(input_text,related_products,user_intent,store_name,store_description):
    try:
        prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a {store_name}'s AI assistant. Your role is to help customers find products, suggest relevant items based on their needs, and provide key details like brand, features, or availability. Respond in 1-2 short, direct sentences (max 20 tokens) with no technical formatting, explanations, or symbols. Avoid preambles and talk in a friendly manner.
        Actual user intention: {user_intent}
        Use related products from: {related_products}."""
    ),
    ("human", "{input}")
])

        chain = prompt | llm

        response = chain.invoke({"input": input_text, "related_products":related_products,"user_intent":user_intent,"store_name":store_name})

        return response.content
    
    except Exception as e:
        print(f"Error in get_response: {str(e)}")
        raise 

def General_QA(query,store_name,store_description):
    try:
        prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a friendly {store_name} AI assistant. Answer any user query—about products, the store, or general topics—in 1-2 short, warm sentences. Avoid preambles, keep it relevant to the store when possible, and redirect off-topic queries with a helpful nudge. Don’t say "sure" first.
        store description : {store_description}

        Examples:
        Query: "What time do you open?" → "We open at 9 AM—come shop soon!"
        Query: "How’s the weather today?" → "Not sure about the weather, but we’ve got umbrellas in stock!"
        Query: "Where are you located?" → "We’re at 123 Main St.—stop by anytime!"
        Query: "What’s a good gift idea?" → "A gift card or stylish scarf makes a great gift!"
        Query: "Do you have a showroom?" → "Yes, visit us at 123 Main St. to see our products."
        Query: "How do I apply for a job at your store?" → "Check our Careers page to apply."
        Query: "Do you offer repairs?" → "Yes, see our Services page for repair options."
        Query: "Do you have a contact phone number?" → "Reach us at 555-123-4567."
        Query: "Where can I find promotions?" → "Visit our Promotions page for current deals."
        Query: "What if I return a defective item?" → "Bring it to the store for a quick exchange."
        Query: "Do you deliver?" → "We do—check our Delivery page for details."
        Query: "Do you alter clothing?" → "Yes, alterations take about a week."
        Query: "Can I rent equipment?" → "We offer rentals—see our Rental page."
        Query: "What credit cards do you accept?" → "We accept Visa, Mastercard, Amex, and Discover."
        Query: "Can I special order an item?" → "Yes, special orders arrive in about 2 weeks."
        Query: "Can I check out the products?" → "Absolutely, come see our selection!"
        Query: "What are your store hours?" → "We’re open Mon-Sat 9 AM-7 PM, closed Sunday."
        """
    ),
    ("human", "{input}")
])

        chain = prompt | llm

        response = chain.invoke({"input": query})

        return response.content
    
    except Exception as e:
        print(f"Error in get_response: {str(e)}")
        raise 

def Store_QA(query,store_name,store_description):

    try:        

        prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a friendly, expert agent at a {store_name} , offering a wide range of top-tier products for all customers. Promote our industry-best products and service, praising customers’ great choice in shopping with us. Answer in 1-2 short sentences (max 20 tokens), using store website links if relevant. Don’t mention competitors or unrelated resources; if unsure, direct to store phone, email, or address (include for location or contact queries). Avoid guesses, ask follow-ups if needed, and keep it accurate. Output should be a single line response.
        store description : {store_description}

        Use the following pieces of context to answer the user's question:
        ----------------
        """
    ),
    ("human", "{input}")
])


        chain = prompt | llm

        response = chain.invoke({"input": query})

        return response.content
    
    except Exception as e:
        print(f"Error in get_response: {str(e)}")
        raise 



def check_want_ask_question(input_text,user_intent,related_products,chat_history,store_name,store_description):
    llm = ChatGoogleGenerativeAI(
    model=model_1,
    temperature=0.7,
    max_tokens=60,
    timeout=None,
    max_retries=2,
    google_api_key=GOOGLE_API_KEY
    )

    try:
        prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a smart decision-making expert for a {store_name} . Analyze the full chat history to determine if the current topic is a scenario-based query (describing a specific situation or task requiring analysis or advice). Return 'YES' if the latest input contributes to or refines a scenario-based topic, 'NO' if it seeks definitions, lists, or unrelated information. Consider prior context to maintain topic continuity.

        Examples:
        1. History:
           User: 'I need to fix a leaky faucet. What tools?' Bot: 'Pipe wrench or sealant?' User: 'Hand tools, simple.'
           Latest Input: 'Can you suggest a brand?'
           Output: YES
        2. History:
           User: 'What is a hammer?' Bot: 'A tool for driving nails.' User: 'What’s a wrench?'
           Latest Input: 'Define a screwdriver.'
           Output: NO
        3. History:
           User: 'I’m buying a dress for a wedding.' Bot: 'Formal or casual?' User: 'Formal, red, size medium.'
           Latest Input: 'Under $100, please.'
           Output: YES
        4. History:
           User: 'I need running shoes for a marathon.' Bot: 'Men’s or women’s?' User: 'Men’s, size 10.'
           Latest Input: 'Nike, black.'
           Output: YES
        5. History:
           User: 'I’m baking a vegan cake.' Bot: 'Need flour or sugar?' User: 'Gluten-free flour.'
           Latest Input: 'What’s gluten-free flour?'
           Output: NO
        6. History:
           User: 'I need organic apples for a recipe.' Bot: 'Fuji or Granny Smith?' User: 'Fuji, 2 pounds.'
           Latest Input: 'List types of apples.'
           Output: NO

        user intention: {user_intent}
        related products: {related_products}
        chat history: {chat_history}

        Analyze the conversation and return 'YES' or 'NO'."""
    ),
    ("human", "{input}")
])
        
        chain = prompt | llm

        response = chain.invoke({"chat_history":chat_history,"input": input_text, "related_products":related_products,"user_intent":user_intent})

        print("\n",response.content)

        return response.content.strip()
    except Exception as e:
        print(f"Error in check_want_ask_question : {str(e)}")
        raise


def ask_question(chat_history,input_text,user_intent,related_products,store_name,store_description):
    llm = ChatGoogleGenerativeAI(
    model=model_1,
    temperature=0.7,
    max_tokens=60,
    timeout=None,
    max_retries=2,
    google_api_key=GOOGLE_API_KEY
    )

    try:
        prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You're a {store_name}'s AI assistant. For scenario-based queries, ask a short, friendly question (max 20 tokens) to help users find products. Use: intent '{user_intent}', products '{related_products}', history '{chat_history}'. Don’t repeat questions. Ensure question fits intent.

        store description : {store_description}

        Categories & Questions:
        - Product Search: 'What type are you looking for?' | 'Any specific features needed?' | 'What size or color?' | 'For what occasion or use?'
        - Pricing: 'What's your budget?' | 'Interested in deals or offers?'
        - Technical Assistance: 'Need help comparing brands?' | 'Unsure which option is best?'

        Examples:
        1. Input: 'I need to fix a leaky pipe.' Intent: 'fixing plumbing issues' Products: 'pipe wrench, sealant, tape' History: '' → 'What type of plumbing tools?'
        2. Input: 'I want to repair my bathroom sink.' Intent: 'fixing plumbing issues' Products: 'pipe wrench, sealant, faucet' History: 'AI: What type of plumbing tools? User: Hand tools.' → 'Which brand do you prefer?'
        3. Input: 'I'm buying a dress for a wedding.' Intent: 'wedding attire' Products: 'evening dress, heels, accessories' History: '' → 'What size or color dress?'
        4. Input: 'I need running shoes.' Intent: 'athletic footwear' Products: 'sneakers, insoles' History: '' → 'What size or brand?'
        5. Input: 'I'm making a vegan cake.' Intent: 'vegan baking' Products: 'flour, sugar, vegan butter' History: '' → 'Need gluten-free or organic?'
        6. Input: 'I need organic apples.' Intent: 'organic groceries' Products: 'apples, bananas' History: 'AI: Need Fuji or Granny Smith? User: Fuji.' → 'How many pounds?'"""
    ),
    ("human", "{input}")
])

        chain = prompt | llm

        response = chain.invoke({"chat_history":chat_history,"input": input_text, "related_products":related_products,"user_intent":user_intent,"store_name":store_name,"store_description":store_description})	

        print("\n",response.content)

        return response.content.strip()
    except Exception as e:
        print(f"Error in research_intent: {str(e)}")
        raise  



   
@app.route('/chat', methods=['POST'])
def chat_product_search():
    try:

        message = request.json
        store_id = message.get('store_id')
        store_name = message.get('store_name', 'Our Store')
        store_description = message.get('description', 'an e-commerce store')
        email = message.get('email')

        if email is None:
            return jsonify({'error': 'email is required'}), 400
        
        query = analyze_intent(message.get('content'),store_name,store_description).lower()
        prioritize_products_response = None

        if query == "general":
            ai_response = General_QA(message.get('content'),store_name,store_description)

        elif query == "website/company":
            ai_response = Store_QA(message.get('content'),store_name,store_description)

        else:

            query = {"Email": email} if email else {"Email": "guest_69dd2db7-11bf-49cc-934c-14fa2811bb4c"}
            chat_history = list(chat_history_collection.find(query))
            # Extract just sender and text from chat history
            chat_history = [{'sender': msg['sender'], 'text': msg['text']} 
                    for chat_doc in chat_history 
                    for msg in chat_doc.get('messages', [])]
            

            if len(chat_history) > 10:
                chat_history = chat_history[-10:]

            chat_history.append({'sender': message.get('sender'), 'text': message.get('content')})

            message.update({
                'timestamp': datetime.now().isoformat()
            })

            # print(chat_history)

            research_intent_response = research_intent(chat_history,store_name,store_description)
            chat_history = []

            # print("\n\nchat_history : ", chat_history)
            print("\n\nresearch_intent_response : ", research_intent_response)

            
            related_product = get_product_search(research_intent_response)

            # print("related_product : ",related_product)

            prioritize_products_response = prioritize_products(research_intent_response,related_product)
            related_product = ""

            toss = check_want_ask_question(input_text = message['content'],user_intent = research_intent_response,related_products=related_product,chat_history=chat_history,store_name=store_name,store_description=store_description)
            
            if toss == 'YES':
                ai_response = ask_question(chat_history = chat_history,input_text = message['content'], user_intent = research_intent_response,related_products=related_product,store_name=store_name,store_description=store_description)
                prioritize_products_response = ""

            else:
                ai_response = get_response(input_text = message['content'], user_intent = research_intent_response,related_products=prioritize_products_response,store_name=store_name,store_description=store_description)
        

        response = {
            'content': ai_response,
            'sender': 'bot',
            'timestamp': datetime.now().isoformat(),
            'related_products_for_query':prioritize_products_response
        }        
        ai_response = ""
        return jsonify(response)
    
    except Exception as e:
        error_response = {
            "error_response" : str(e),
            'content': "I apologize, but I encountered an error. Please try again.",
            'sender': 'bot',
            'error': True,
            'timestamp': datetime.now().isoformat()
        }
        return jsonify(error_response), 500

    
@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "working"})

if __name__ == "__main__":
    app.run(debug=False)