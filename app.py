from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from config import Config
from models import db, Document
import faiss
import requests
import time
from langdetect import detect
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import asyncio
import aiohttp
import numpy as np
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
import redis
import json
import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel
import torch
from sentence_transformers import SentenceTransformer
import os
from transformers import AutoTokenizer
import fitz  # PyMuPDF
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)
app.config['UPLOAD_FOLDER'] = 'documents'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

db.init_app(app)

HF_API_TOKEN = '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'
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2', token=HF_API_TOKEN)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def export_and_optimize_model():
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    text = "This is a dummy input to trace the model."
    inputs = tokenizer(text, return_tensors="pt")
    
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    class WrappedModel(torch.nn.Module):
        def __init__(self, model):
            super(WrappedModel, self).__init__()
            self.model = model

        def forward(self, input_ids, attention_mask):
            return self.model({'input_ids': input_ids, 'attention_mask': attention_mask})['sentence_embedding']
    
    wrapped_model = WrappedModel(model)
    
    torch.onnx.export(
        wrapped_model,
        (input_ids, attention_mask),
        "model.onnx",
        input_names=['input_ids', 'attention_mask'],
        output_names=['sentence_embedding'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
            'sentence_embedding': {0: 'batch_size', 1: 'embedding_size'}
        }
    )
    
    options = SessionOptions()
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    onnx_session = InferenceSession("model.onnx", options)
    
    return onnx_session

if not os.path.exists("model.onnx"):
    onnx_session = export_and_optimize_model()
else:
    options = SessionOptions()
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    onnx_session = InferenceSession("model.onnx", options)

d = 384
faiss_index = faiss.IndexHNSWFlat(d, 32)
faiss_index.hnsw.efConstruction = 40

r = redis.Redis(host='localhost', port=6379, db=0)

def cache_set(key, value, ttl=300):
    r.setex(key, ttl, json.dumps(value))

def cache_get(key):
    value = r.get(key)
    if value:
        return json.loads(value)
    return None

es = Elasticsearch(hosts=["http://localhost:9200"])

def index_documents(docs):
    actions = [
        {
            "_index": "documents",
            "_id": doc.id,
            "_source": {
                "title": doc.title,
                "content": doc.content,
                "reference": doc.reference,
                "language": doc.language
            }
        }
        for doc in docs
    ]
    bulk(es, actions)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
async def query():
    data = request.json
    query_text = data['query']
    query_language = detect(query_text)
    
    print(f"Received query: {query_text} in language: {query_language}")
    
    cached_result = cache_get(query_text)
    if cached_result:
        return jsonify(cached_result)
    
    input_features = tokenizer(query_text, return_tensors='pt', padding=True, truncation=True)
    inputs_onnx = {k: v.numpy() for k, v in input_features.items() if k != 'token_type_ids'}
    query_embedding = await asyncio.to_thread(onnx_session.run, None, inputs_onnx)
    
    query_embedding = np.array(query_embedding[0])
    if len(query_embedding.shape) == 1:
        query_embedding = query_embedding.reshape(1, -1)
    
    print(f"Query embedding: {query_embedding}")
    
    try:
        D, I = faiss_index.search(query_embedding, k=5)
        print(f"Search results indices: {I}")
    except Exception as e:
        print(f"Error searching FAISS index: {e}")
        return jsonify({"error": "Error searching FAISS index"}), 500
    
    results = []
    for idx in I[0]:
        doc_id = int(idx)
        print(f"Document ID: {doc_id}")
        doc = Document.query.get(doc_id)
        if doc:
            results.append({
                'title': doc.title,
                'content': doc.content,
                'reference': doc.reference,
                'language': doc.language
            })
    print(f"Retrieved documents: {results}")

    try:
        answer = await generate_answer(query_text, results, query_language)
        print(f"Generated answer: {answer}")
    except Exception as e:
        print(f"Error generating answer: {e}")
        return jsonify({"error": "Error generating answer"}), 500
    
    cache_set(query_text, answer)
    
    return jsonify(answer)

@app.route('/add_document', methods=['POST'])
async def add_document():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        text = extract_text_from_pdf(file_path)
        language = detect(text)  # Detect language from the extracted text

        new_doc = Document(
            title=filename,
            content=text,
            reference=file_path,
            language=language
        )
        db.session.add(new_doc)
        db.session.commit()

        # Batch processing for document embedding
        batch_size = 32
        documents = [new_doc.content]
        embeddings = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            tokenized_batch = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=128)
            input_ids = tokenized_batch['input_ids'].numpy()
            attention_mask = tokenized_batch['attention_mask'].numpy()
            batch_embeddings = await asyncio.to_thread(onnx_session.run, None, {'input_ids': input_ids, 'attention_mask': attention_mask})
            embeddings.extend(batch_embeddings[0])

        embeddings = np.array(embeddings)
        faiss_index.add(embeddings)
        print(f"FAISS index size after adding document: {faiss_index.ntotal}")

        index_documents([new_doc])

        return jsonify({'message': 'Document added successfully'}), 201

    return jsonify({'error': 'Invalid file type'}), 400


@lru_cache(maxsize=1000)
def get_document_embedding(content):
    tokenized_content = tokenizer([content], return_tensors='pt', padding=True, truncation=True, max_length=128)
    input_ids = tokenized_content['input_ids'].numpy()
    attention_mask = tokenized_content['attention_mask'].numpy()
    return onnx_session.run(None, {'input_ids': input_ids, 'attention_mask': attention_mask})[0]

async def generate_answer(query, docs, query_language):
    context = " ".join([doc['content'] for doc in docs if doc['language'] == query_language])
    prompt = f"Q: {query}\nContext: {context}\nA:"

    url = 'http://localhost:11435/api/generate'
    payload = {
        "model": "llama3",
        "prompt": prompt,
        "stream": False
    }
    
    start_time = time.time()
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            response_data = await response.json()
            print(f"Response data: {response_data}")
    
    end_time = time.time()
    response_time = end_time - start_time
    
    if 'response' in response_data:
        answer = response_data['response']
    else:
        print("Response JSON does not contain 'response' key.")
        answer = "No answer available."
    
    references = [doc['reference'] for doc in docs if doc['language'] == query_language]
    
    return {"answer": answer, "references": references, "response_time": response_time}

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        
        # Initialize FAISS index with existing documents
        docs = Document.query.all()
        if docs:
            embeddings = []
            batch_size = 32
            for i in range(0, len(docs), batch_size):
                batch_docs = docs[i:i + batch_size]
                tokenized_batch = tokenizer([doc.content for doc in batch_docs], return_tensors='pt', padding=True, truncation=True, max_length=128)
                input_ids = tokenized_batch['input_ids'].numpy()
                attention_mask = tokenized_batch['attention_mask'].numpy()
                batch_embeddings = onnx_session.run(None, {'input_ids': input_ids, 'attention_mask': attention_mask})
                embeddings.extend(batch_embeddings[0])
            embeddings = np.array(embeddings)
            
            # Adjust number of clusters based on the number of documents
            if len(embeddings) > 100:
                nlist = min(len(embeddings), 100)
                quantizer = faiss.IndexFlatL2(384)
                faiss_index = faiss.IndexIVFFlat(quantizer, 384, nlist, faiss.METRIC_L2)
                faiss_index.train(embeddings)
            else:
                faiss_index = faiss.IndexFlatL2(384)

            faiss_index.add(embeddings)
            print(f"FAISS index initialized with {len(docs)} documents")

    app.run(debug=True)