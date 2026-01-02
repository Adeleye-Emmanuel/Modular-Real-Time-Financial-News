import re
import numpy as np
import pandas as pd
from src.scraper import Scraper

def clean_corpus(text):

    """
    Comprehensive text cleaning function that is built to handle
    - Newlines and tabs
    - Irrelevant prefix and suffix (e.g skip comments)
    - Javascript snippets
    - URLs
    - Special characters and excessive whitespace
    - Short sentences
    """
    if not isinstance(text,str):
        return ""

    # Remove newlines, tabs, and excessive whitespaces
    text = " ".join(text.split())

    # Removing javascript snippets and HTML tags
    text = re.sub(r'{.*?}', "", text)
    text = re.sub(r'href.*?\)', "", text)
    text = re.sub(r'<.*?>', "", text)

    # Removing "skip to comments" and similar patterns
    text = re.sub(r'skip to comments.*?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Click here to view the full post.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r"click 'Accept all'.*", '', text, flags=re.IGNORECASE)

    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)

    # Remove special characters
    text = re.sub(r"""[^\w\s.,;:!?'"-]""", '', text)

    # Remove standalone single/double quotes
    text = re.sub(r'\s[\'"]\s',' ', text)

    # Remove trailing/leading whitespaces
    text = text.strip()

    return text

def refine_corpus(corpus, min_length=70):

    """
    Refining corpus by splitting into robust sentences, applying comprehensive text cleaning and filtering the corpus length
    """

    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', corpus)

    cleaned_sentences = []
    for sentence in sentences:
        # Cleaning corpus
        clean_sent = clean_corpus(sentence)

        if len(clean_sent) >= min_length:
            # Ensuring sentences end with a proper punctuation
            if not clean_sent.endswith(('.','?','!')):
                clean_sent +='.'
            cleaned_sentences.append(clean_sent)    
    
    return  cleaned_sentences

def news_pull(s_query, s_type, include_media, media_percent=None):
    scraper = Scraper(s_query)
    if include_media==True:
        if isinstance(media_percent, float):
            if s_type == 'general': # specifying general drops ticker based search from alphaadvantage
                napi_df = scraper.fetch_newsapi(s_query, 100, "full")
                aapi_df = None
                reddit_df = scraper.fetch_reddit(query=s_query, percent=media_percent)
                rss_df = scraper.fetch_multiple_rss(query=s_query)
            
            elif s_type == 'ticker': # specifying ticker includes alphaadvantage ticker based news
                napi_df = scraper.fetch_newsapi(s_query, 100, "full")
                aapi_df = scraper.fetch_alpha_vantage(s_query)
                reddit_df = scraper.fetch_reddit(query=s_query, percent=0.3)
                rss_df = scraper.fetch_multiple_rss(query=s_query)

        else:
            print('Enter a valid percentage value for media articles')
    
    elif include_media==False:
        if s_type == 'general':
            napi_df = scraper.fetch_newsapi(s_query, 100, "full")
            aapi_df = None
            reddit_df = None
            rss_df = scraper.fetch_multiple_rss(query=s_query)
        
        elif s_type == 'ticker':
            napi_df = scraper.fetch_newsapi(s_query, 100, "full")
            aapi_df = scraper.fetch_alpha_vantage(s_query)
            reddit_df = None
            rss_df = scraper.fetch_multiple_rss(query=s_query)  
    
    dfs = [df for df in [napi_df, aapi_df, reddit_df, rss_df] if df is not None]
    full_response = pd.concat(dfs, axis=0).reset_index(drop=True)
    full_response['full_response'] = " Source: " + full_response['source'] + "\n" + full_response['title'] + '\n' + full_response['content']
    
    texts_list = full_response['title'] + '\n' + full_response['content']

    # Stack them all into a single string
    all_text = "\n".join(texts_list)                             
    return full_response, all_text