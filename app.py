import os
import time
import numpy as np
import tqdm
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from pymongo.operations import UpdateOne
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from multiprocessing import Pool, cpu_count
from pinecone import Pinecone, ServerlessSpec

# Environment variables
hf_token = os.getenv('HF_TOKEN', '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')
Pinecone_api_key = os.getenv('PINECONE_API_KEY', 'pcsk_3XafLm_SDzfsZm5fmrpXPafmpaUaJydXGr4KucreMZpGay5Uz84MAY4mk9tKYqKTeNUMrp')
MONGO_URI = os.getenv('MONGO_URI', 'mongodb+srv://jsckson_store:jsckson_store@cluster0.9a981.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')

# Single MongoDB client
client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize model
model = HuggingFaceInferenceAPIEmbeddings(api_key=hf_token, model_name="sentence-transformers/all-MiniLM-L6-v2")

# Global Pinecone index
Pinecone_index = None

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

def parallel_embed(documents, batch_size=32, num_processes=cpu_count()):
    batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]
    print(f"Processing {len(batches)} batches with {num_processes} processes")
    with Pool(processes=num_processes) as pool:
        results = list(tqdm.tqdm(pool.imap(generate_embedding_batch, batches), total=len(batches)))
    return [item for sublist in results for item in sublist]

def build_embedding(database_name, collection_name):
    db = client[database_name]
    collection = db[collection_name]
    
    # Process documents in chunks
    chunk_size = 1000
    total_docs = collection.count_documents({})
    embeddings = []
    
    for skip in range(0, total_docs, chunk_size):
        documents = list(collection.find().skip(skip).limit(chunk_size))
        print(f"Processing documents {skip + 1} to {skip + len(documents)}")
        batch_embeddings = parallel_embed(documents, batch_size=32, num_processes=cpu_count())
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
    
    print("Embeddings generated and stored in MongoDB!")

def build_pinecone_vectorstore(database_name, collection_name):
    db = client[database_name]
    collection = db[collection_name]
    
    pc = Pinecone(api_key=Pinecone_api_key)
    index_name = collection_name
    
    if index_name in pc.list_indexes().names():
        try:
            pc.delete_index(index_name)
        except Exception as e:
            print(f"Error deleting Pinecone index: {e}")
    
    for _ in range(3):
        try:
            pc.create_index(
                name=index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
            break
        except Exception as e:
            print(f"Error creating Pinecone index: {e}")
            time.sleep(2)
    
    index = pc.Index(index_name)
    
    # Process documents in chunks
    chunk_size = 1000
    total_docs = collection.count_documents({})
    batch_size = 500
    
    for skip in range(0, total_docs, chunk_size):
        documents = collection.find().skip(skip).limit(chunk_size).batch_size(batch_size)
        to_upsert = []
        
        for doc in documents:
            if 'embedding' not in doc or not doc['embedding']:
                print(f"Skipping document {doc['_id']}: Missing or invalid embedding")
                continue
            id_value = f"id_{str(doc['_id'])}"
            embedding = doc['embedding']
            metadata = {k: v for k, v in doc.items() if k not in ['_id', 'embedding']}
            metadata = {
                k: "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)
                for k, v in metadata.items()
            }
            to_upsert.append({"id": id_value, "values": embedding, "metadata": metadata})
            
            if len(to_upsert) >= batch_size:
                try:
                    index.upsert(vectors=to_upsert)
                    to_upsert = []
                except Exception as e:
                    print(f"Error upserting batch to Pinecone: {e}")
        
        if to_upsert:
            try:
                index.upsert(vectors=to_upsert)
            except Exception as e:
                print(f"Error upserting final batch to Pinecone: {e}")
    
    global Pinecone_index
    Pinecone_index = index
    print("Pinecone index reset and data upserted successfully!")

def get_recommendations(query, top_k=30):
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
        if database_name not in client.list_database_names():
            return jsonify({"error": f"Database {database_name} does not exist"}), 400
        db = client[database_name]
        if collection_name not in db.list_collection_names():
            return jsonify({"error": f"Collection {collection_name} does not exist"}), 400
    except Exception as e:
        return jsonify({"error": f"Error validating MongoDB: {str(e)}"}), 500
    
    try:
        build_embedding(database_name, collection_name)
    except Exception as e:
        return jsonify({"error": f"Error in build_embedding: {str(e)}"}), 500
    
    try:
        build_pinecone_vectorstore(database_name, collection_name)
    except Exception as e:
        return jsonify({"error": f"Error in build_pinecone_vectorstore: {str(e)}"}), 500
    
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

@app.route("/")
def health_check():
    return jsonify({"status":"working"}),200

if __name__ == '__main__':
    app.run()
