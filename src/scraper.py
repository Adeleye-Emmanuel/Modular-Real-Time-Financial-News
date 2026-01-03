import requests
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from datetime import datetime
import re
import feedparser
import praw
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

class Scraper:

    def __init__(self, query):
        load_dotenv()
        self.news_api_key = os.getenv("NEWS_API_KEY")
        self.news_api_url = "https://newsapi.org/v2/everything"
        self.av_api_url = "https://www.alphavantage.co/query"
        self.news_apikey = os.getenv("NEWS_API_KEY")
        self.alpha_vantage_apikey = os.getenv("ALPHA_VANTAGE_API_KEY")
        self.openai_api = os.getenv("openai_api")
        self.client_secret = os.getenv("client_secret")
        self.client_id = os.getenv("client_id")
        self.user_agent = os.getenv("user_agent")
        self.cohere_api = os.getenv("cohere_api")
        self.query = query
    
    def _clean_results_napi(self, content):
        if not content:
            return ""

        content = re.sub(r'\[\+\d+ chars\]',"",content)    
        content = '\n'.join(line.strip() for line in content.splitlines() if line.strip())

        return content.strip()
    
    def fetch_newsapi(self,query, num_articles, content_type):
        # The function takes in the users query they intend to analyse for, the number of articles and the content type, that is either full, partial or none
        url = self.news_api_url
        params = {
            "q":query,
            "apiKey":self.news_apikey,
            "pageSize":num_articles,
            "sortBy":"publishedAt",
            "content":content_type
        }
        response = requests.get(url, params=params).json()
        articles = response.get("articles", [])
        parsed_articles = [
            {
            "title": a["title"], 
            "content":a.get("content") or a.get("description", ""), # Fallback 
            "source": "NewsAPI", 
            "date": a["publishedAt"]
        } 
                for a in articles]
        
        df = pd.DataFrame(parsed_articles)
        df['content'] = df['content'].apply(self._clean_results_napi)

        return df
    
    def fetch_alpha_vantage(self):
        params = {
        "function":"NEWS_SENTIMENT",
        "tickers":self.query,
        "apikey":self.alpha_vantage_apikey,
        "sort": "LATEST"
    }

        try:
            response = requests.get(self.av_api_url, params=params)
            response.raise_for_status() # this will raise http errors
            data = response.json()
            articles = data.get("feed", [])

            
            parsed_articles = []
            for article in articles:
                # Parse date from "20250121T120000" to datetime
                pub_date = datetime.strptime(article['time_published'], "%Y%m%dT%H%M%S")

                # Getting sentiment scores for the queried tickers(s)
                ticker_sentiments = [
                    ts for ts in article.get("ticker_sentiment", [])
                    if ts['ticker'] in self.query.split(",")
                ]

                parsed_articles.append({
                    "title":article['title'],
                    "content":article.get("summary", ""),
                    "source":"AlphaVantage",
                    "date":pub_date,
                    "url":article.get("url", ""),
                    "sentiment_label":article.get("overall_sentiment_label", "neutral"),
                    "relevance_score": float(article.get("relevance_score", 0)),
                    "ticker_sentiment": ticker_sentiments
                    
                })
            
            df = pd.DataFrame(parsed_articles)
            
            return df
        
        except Exception as e:
            print(f"Error fetching data from Alpha Vantage: {e}")
            return pd.DataFrame()
    
    def _fetch_rss_feed(self, feed_url, num_articles):
        feed = feedparser.parse(feed_url)    

        articles = []

        for entry in feed.entries:
            
            # Safely get title and content (summary/description)
            title = entry.get('title', '')
            summary = entry.get('summary', entry.get('description', ''))

            # Checking if the query matches the title or summary
            if self.query and self.query.lower() not in (title + summary).lower():
                continue # this skips non-matching articles
            
            # Parse the publication date
            pub_date = datetime(*entry.published_parsed[:6]) if hasattr(entry, "published_parsed") else None

            articles.append({
                "title": entry.title,
                "content": entry.summary,
                "source": feed_url,
                "date": pub_date,
                "url": entry.link
            })
        
        # sort articles by date, most recent first
        articles = sorted(articles, key=lambda x: x['date'] or datetime.min, reverse=True)
        if num_articles < len(articles):
            articles = articles[:num_articles]
        else:
            articles = articles
        return articles

    def fetch_multiple_rss(self, query):
        all_articles = []
        feed_urls = [
                "https://feeds.bloomberg.com/technology/news.rss",
                "https://feeds.bloomberg.com/markets/news.rss",
                "https://feeds.bloomberg.com/politics/news.rss",
                "https://feeds.bloomberg.com/businessweek/news.rss",
                "https://feeds.bloomberg.com/economics/news.rss",
                "https://feeds.bloomberg.com/industries/news.rss",
                "https://feeds.bloomberg.com/bview/news.rss",
                "https://feeds.bloomberg.com/wealth/news.rss"
                    ]
        for feed_url in feed_urls:
            articles = self._fetch_rss_feed(feed_url, num_articles=100)
            all_articles.extend(articles)
        
        df = pd.DataFrame(all_articles)
        return df
    
    def fetch_reddit(self, query, percent, subreddit='wallstreetbets', limit=100):
        

        reddit = praw.Reddit(
            client_id=self.client_id,
            client_secret=self.client_secret,
            user_agent=self.user_agent
        )

        subreddit = reddit.subreddit(subreddit)
        posts = subreddit.search(query, limit=limit, sort='new')

        articles = []
        for post in posts:
            articles.append({
                "title": post.title,
                "content": post.selftext,
                "source": f"Reddit r/{subreddit}",
                "date": datetime.fromtimestamp(post.created_utc),
                "url": f"https://www.reddit.com{post.permalink}",
                "upvote_ratio": post.upvote_ratio,
                "score": post.score,
                "num_comments": post.num_comments
            })
        
        df = pd.DataFrame(articles)

        threshold = np.percentile(df['score'], percent)
        df_filtered = df[df['score'] >= threshold].reset_index(drop=True)

        return df_filtered