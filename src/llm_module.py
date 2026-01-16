import requests
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from datetime import datetime
import time

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

import cohere
import faiss
from tqdm import tqdm
import random

import praw
import re
from datasets import Dataset
import sys
from src.utils import clean_corpus, refine_corpus
from src.scraping_module import *

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction import _stop_words
import string

from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

import openai

load_dotenv()
cohere_api = os.getenv("cohere_api")
co = cohere.Client(cohere_api)
openai_api = os.getenv("openai_api")

response_schemas = [
    ResponseSchema(name="key_insights", description='3-5 bullet points summarizing key insights/outlook on the topic'),
    ResponseSchema(name="key_drivers", description='Main economic/politcal indicators driving the topic'),
    ResponseSchema(name='risks', description='Potential risks associated with the topic'),
    ResponseSchema(name='sentiment', description='Overall social sentiment (positive/negative/neutral with evidence) and degree of sentiment in percentage')    
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

def analyze_text(s_query, relevant_text):
    
    # initiate llm model
    prompt = ChatPromptTemplate.from_template(
    """
    Analyze the following news corpus regarding {query} 
    ### RULES:
        1. ONLY use the provided Corpus to answer. 
        2. If the Corpus does not contain information for a specific field, return "Not specified in corpus" for that field. 
        3. DO NOT use your internal training data to invent risks or drivers.
        4. For 'sentiment', ensure the percentage is derived directly from context or tone indicators in the text.
    
    and extract:
    {format_instructions}
    
    Corpus:
    {text}
    
    After generating the analysed results, loop back again to verify all instructions and rules are followed and if not adjust to comply.
    """)
    #client = openai.OpenAI(api_key=openai_key)
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.3,
        openai_api_key=openai_api  # Pass key directly or use environment variable
    )

    messages = prompt.format_messages(
        query = s_query,
        text = relevant_text,
        format_instructions=format_instructions
    )

    response = llm(messages)
    return output_parser.parse(response.content)

def create_search_index(full_text):
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=120
    )
    chunks = text_splitter.split_text(full_text)

    # creating searchable index
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api)
    return FAISS.from_texts(chunks, embeddings)    

def analyze_with_semantic_search(s_query, text_list, n_results=8):
    full_texts = " ".join(text_list) if isinstance(text_list, list) else text_list
    # creating vector index on full corpus
    index = create_search_index(full_texts)

    # retreiving relevant chunks
    bm25_retriever = BM25Retriever.from_texts(full_texts)
    faiss_retriever = index.as_retriever()
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever], 
        weights=[0.5,0.5]
    )
    
    relevant_text = ensemble_retriever.get_relevant_documents(s_query)
    relevant_text = relevant_text[:n_results] if len(relevant_text)>n_results else relevant_text
    
    return analyze_text(s_query, relevant_text), relevant_text