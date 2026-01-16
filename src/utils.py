import re
import numpy as np
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

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