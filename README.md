# Modular Real-Time Modeling for Financial Insight Generation

## 🧾 Overview

This project presents a modular, real-time financial news intelligence system that aggregates data from multiple sources — including **NewsAPI**, **Alpha Vantage**, **Bloomberg RSS feeds**, and **Reddit** — and uses **Cohere embeddings**, **FAISS indexing**, and **LLM-based analysis** to extract actionable insights from live market narratives.

The goal is to help analysts, retail investors, or research teams **query the latest market sentiment, policy effects, or investment outlooks** in real time. By integrating vector similarity, keyword matching, and semantic reranking, this system goes beyond keyword-based news aggregation and into **true insight generation.**

It supports:
- 🔎 Real-time semantic search
- 📈 Insight and risk summarization
- 💬 Sentiment analysis
- 🧠 Multi-source context fusion (Reddit, RSS, AlphaVantage, NewsAPI)

## 🚧 Problem Statement

With financial news scattered across platforms and formats, analysts face significant friction in extracting insights efficiently:
- News APIs return generic articles with no deep filtering
- Reddit and RSS feeds offer raw opinions and policy reactions
- Insight often requires synthesis across **structured and unstructured** sources

This project solves that by:
- Creating a **modular ingestion framework**
- Indexing and embedding content using **Cohere + FAISS**
- Performing **semantic retrieval, keyword search, and reranking**
- Summarizing results via **LLM-based insight parsing** using LangChain and GPT-3.5

The result is a unified system that answers investor-style questions like:
> *"What are the current risks associated with U.S. tariffs on China?"*  
> *"What are Reddit users saying about Nvidia’s earnings?"*

## 🛠️ Modular Pipeline

The project is fully modular and can run in standalone or integrated mode:

### 📥 Data Sources:
- **NewsAPI**: General article search
- **Alpha Vantage**: Ticker-based sentiment feeds
- **Bloomberg RSS Feeds**: Real-time macroeconomic headlines
- **Reddit (r/WallStreetBets)**: Public discourse + upvote-based filtering

### 🧹 Preprocessing:
- Regex-based corpus cleaner
- Sentence-level filtering and punctuation correction
- Length-based sentence pruning

### 🧠 Embedding & Indexing:
- `co.embed()` via Cohere for document/query vectors
- **FAISS** index for vector-based semantic search
- **BM25** lexical keyword search and reranking

### 🔎 Search Modes:
- **Option 1**: Vector search only (via Cohere + FAISS)
- **Option 2**: Hybrid BM25 keyword + semantic reranking
- **Option 3**: LangChain + OpenAI ensemble retriever with chunked document querying

### 💬 Insight Generation:
- Prompted LLMs (GPT-3.5) return structured summaries:
  - **Key Insights**
  - **Key Drivers**
  - **Risks**
  - **Overall Sentiment**

## 🧪 Example Results

Sample Query: **"Impact of tariffs"**
Generated Result:
{'key_insights': "1. War with top trade partners impacting investor risk appetite. 2. Cryptocurrency trading in lockstep with stocks indicating uncertainty for the US currency. 3. European stocks rise on hopes of tariff pause by Trump administration. 4. NFT marketplace gaining traction as a one-stop shop for digital items. 5. Markets reacting to Trump's tariff war with caution.", 
'key_drivers': "Trade tensions with top partners, Trump administration's tariff policies, cryptocurrency market behavior, European stock market trends, NFT marketplace adoption.", 'risks': 'Potential risks include increased market volatility, uncertainty in currency markets, trade war escalation impacting global economy, and regulatory challenges for NFT marketplace.', 
'sentiment': 'Overall sentiment is neutral with a slight negative bias due to concerns over trade tensions and market volatility. Degree of sentiment is 60% negative based on the cautious market behavior and uncertainties highlighted.'}
> 🔁 Multiple runs can compare Reddit-included vs. professional-only sources.

## 🎯 Key Features

- 🔄 **Multi-source Ingestion**: NewsAPI, AlphaVantage, RSS, Reddit
- 🧼 **Robust Cleaning**: Tailored preprocessing for unstructured media
- 🔍 **Semantic Search Engine**: FAISS + Cohere + BM25 hybrid
- 🧠 **LLM-Driven Insight Extraction**: LangChain + GPT-3.5 summaries
- ⚡ **Fast Modular Setup**: Switch search methods or sources on-the-fly
- 📊 **Query Tracking**: Test with or without social discourse (Reddit toggle)
