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
import ntscraper
from newspaper import Article, Config
import requests
from bs4 import BeautifulSoup
import sys
from src.utils import clean_corpus, refine_corpus, news_pull
from src.config import config
from urllib.parse import quote_plus
from datetime import datetime, timedelta
import finnhub

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

class BaseScraper:
    def __init__(self, query, q_type=None, from_date=datetime.now().strftime("%Y-%m-%d"), 
                 to_date=(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")): # q_type could be 'ticker: company_news' or 'general : market_news' add today as to date
        self.query = query
        self.q_type = q_type
        self.from_date = from_date
        self.to_date = to_date

    def fetch_data(self):
        raise NotImplementedError("This method should be overridden by subclasses.")
    
class NewsAPIScraper(BaseScraper):
    def __init__(self, query, num_articles=100, content_type="full"):
        super().__init__(query)
        self.news_apikey = os.getenv("NEWS_API_KEY")
        self.num_articles = num_articles
        self.content_type = content_type
        self.base_url = "https://newsapi.org/v2/everything"
    
    def _clean_results_napi(self, content):
        if not content:
            return ""

        content = re.sub(r'\[\+\d+ chars\]',"",content)    
        content = '\n'.join(line.strip() for line in content.splitlines() if line.strip())

        return content.strip()
    
    def fetch_data(self):
        url = self.base_url
        params = {
            "q": self.query,
            "apiKey": self.news_apikey,
            "pageSize": self.num_articles,
            "sortBy": "publishedAt",
            "content": self.content_type
        }
        try:
            response = requests.get(url, params=params).json()
            articles = response.get("articles", [])
            parsed_articles = [
                {
                    "title": a["title"],
                    "content": a.get("content") or a.get("description", ""),  # Fallback
                    "source": "NewsAPI",
                    "date": a["publishedAt"]
                }
                for a in articles
            ]
            df = pd.DataFrame(parsed_articles)
            df['content'] = df['content'].apply(self._clean_results_napi)
            return df
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching NewsAPI data: {e}")
            return pd.DataFrame()
        
class AlphaVantageScraper(BaseScraper):
    def __init__(self, query):
        super().__init__(query)
        self.alpha_vantage_apikey = os.getenv("ALPHA_VANTAGE_API_KEY")
        self.base_url = "https://www.alphavantage.co/query"
    
    def fetch_data(self):
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": self.query,
            "apikey": self.alpha_vantage_apikey,
            "sort": "LATEST"
        }
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()  # this will raise http errors
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
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching Alpha Vantage data: {e}")
            return pd.DataFrame()
        
class FinnhubScraper(BaseScraper):
    def __init__(self, query, q_type, from_date=None, to_date=None):
        super().__init__(query, q_type, from_date, to_date)

        api_key = os.getenv("FINNHUB_API_KEY")
        if not api_key:
            raise ValueError("FINNHUB_API_KEY not set in environment")

        self.client = finnhub.Client(api_key=api_key)

    def _fetch_company_news(self, num_articles):
        try:
            response = self.client.company_news(
                symbol=self.query,
                _from=self.from_date,
                to=self.to_date
            )

            articles = [
                {
                    "title": a.get("headline", ""),
                    "content": a.get("summary", ""),
                    "source": a.get("source", ""),
                    "date": datetime.fromtimestamp(a["datetime"]).isoformat(),
                    "url": a.get("url", "")
                }
                for a in response
            ][:num_articles]

            return pd.DataFrame(articles)

        except Exception as e:
            logger.error(f"Error fetching Finnhub company news: {e}")
            return pd.DataFrame()

    def _fetch_market_news(self, num_articles):
        try:
            response = self.client.general_news(category="general")

            articles = [
                {
                    "title": a.get("headline", ""),
                    "content": a.get("summary", ""),
                    "source": a.get("source", ""),
                    "date": datetime.fromtimestamp(a["datetime"]).isoformat(),
                    "url": a.get("url", "")
                }
                for a in response
                if self.query.lower() in a.get("headline", "").lower()
            ][:num_articles]

            return pd.DataFrame(articles)

        except Exception as e:
            logger.error(f"Error fetching Finnhub market news: {e}")
            return pd.DataFrame()

    def fetch_data(self):
        if self.q_type == "ticker":
            return self._fetch_company_news(num_articles=50)
        elif self.q_type == "general":
            return self._fetch_market_news(num_articles=50)
        else:
            raise ValueError("q_type must be either 'ticker' or 'general'")

class RSSFeedScraper(BaseScraper):
    def __init__(self, query):
        super().__init__(query, from_date=None, to_date=None)
        self.feed_urls = {
        # Financial News
        'reuters_business': 'https://www.reutersagency.com/feed/?taxonomy=best-topics&post_type=best',
        'reuters_markets': 'http://feeds.reuters.com/reuters/businessNews',
        'ft_companies': 'https://www.ft.com/companies?format=rss',
        'ft_markets': 'https://www.ft.com/markets?format=rss',
        'wsj_markets': 'https://feeds.a.dj.com/rss/RSSMarketsMain.xml',
        'wsj_economy': 'https://feeds.a.dj.com/rss/WSJcomUSBusiness.xml',
        'cnbc': 'https://www.cnbc.com/id/100003114/device/rss/rss.html',
        'marketwatch': 'http://feeds.marketwatch.com/marketwatch/topstories/',
        
        # Tech/Startup News
        'techcrunch': 'https://techcrunch.com/feed/',
        'venturebeat': 'https://venturebeat.com/feed/',
        'theregister': 'https://www.theregister.com/headlines.atom',
        
        # Economic Data
        'fed_news': 'https://www.federalreserve.gov/feeds/press_all.xml',
        'ecb_news': 'https://www.ecb.europa.eu/rss/press.html',
        
        # Sector-Specific
        'seekingalpha': 'https://seekingalpha.com/market_currents.xml',
        'benzinga': 'https://www.benzinga.com/feed',
        
        # Bloomberg Feeds
        'bloomberg_technology': 'https://feeds.bloomberg.com/technology/news.rss',
        'bloomberg_markets': 'https://feeds.bloomberg.com/markets/news.rss',
        'bloomberg_politics': 'https://feeds.bloomberg.com/politics/news.rss',
        'bloomberg_businessweek': 'https://feeds.bloomberg.com/businessweek/news.rss',
        'bloomberg_economics': 'https://feeds.bloomberg.com/economics/news.rss',
        'bloomberg_industries': 'https://feeds.bloomberg.com/industries/news.rss',
        'bloomberg_bview': 'https://feeds.bloomberg.com/bview/news.rss',
        'bloomberg_wealth': 'https://feeds.bloomberg.com/wealth/news.rss',

        # Alternative Data
        'yahoo_finance': 'https://finance.yahoo.com/news/rssindex',
        'investing_com': 'https://www.investing.com/rss/news.rss',
    }
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
    def fetch_data(self, max_articles_per_feed=50):
        all_articles = []
        for name, url in self.feed_urls.items():
            try:
                articles = self._fetch_rss_feed(url, max_articles_per_feed)
                for article in articles:
                    article["source"] = name
                all_articles.extend(articles)
            except Exception as e:
                logger.error(f"Error fetching from {url}: {e}")
                continue
        df = pd.DataFrame(all_articles)
        return df
    
class GoogleNewsScraper(BaseScraper):
    def __init__(self, query):
        super().__init__(query)
    
    def _clean_content(self, text):
        if not text:
            return ""
        text = re.sub(r'\[\+\d+ chars\]','',text)    
        text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
        text = '\n'.join(line.strip() for line in text.splitlines() if line.strip())
        return text.strip()
    
    def fetch_data(self, num_articles=100):
        encoded_query = quote_plus(self.query)
        url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
        
        try:
            feed = feedparser.parse(url)
            
            if not feed.entries:
                logger.warning(f"No Google News results for: {self.query}")
                return pd.DataFrame()
            
            all_articles = []
            for entry in feed.entries[:num_articles]:
                title = entry.get('title', '')
                summary = entry.get('summary', entry.get('description', ''))
                
                # Clean HTML from summary
                clean_summary = self._clean_content(summary)
                
                # Skip if too short
                if len(clean_summary) < 50:
                    continue
                
                # Parse date
                pub_date = None
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    try:
                        pub_date = datetime(*entry.published_parsed[:6])
                    except:
                        pass
                
                # Extract source from title (Google News format: "Title - Source")
                source = "Google News"
                if ' - ' in title:
                    parts = title.rsplit(' - ', 1)
                    if len(parts) == 2:
                        source = parts[1]
                        title = parts[0]
                
                all_articles.append({
                    "title": title,
                    "content": clean_summary,
                    "source": source,
                    "date": pub_date,
                    "url": entry.link
                })
            
            return pd.DataFrame(all_articles)
            
        except Exception as e:
            logger.error(f"Error fetching Google News: {e}")
            return pd.DataFrame()

class RedditScraper(BaseScraper):
    def __init__(self, query):
        super().__init__(query)
        self.client_secret = os.getenv("client_secret")
        self.client_id = os.getenv("client_id")
        self.user_agent = os.getenv("user_agent")
        self.reddit = praw.Reddit(client_id=self.client_id,
                            client_secret=self.client_secret,
                            user_agent=self.user_agent)

    def fetch_data(self, subreddits=None, limit=50):
        """
        Quality subreddits: investing, stocks, SecurityAnalysis, 
        options, algotrading, economics
        """
        if subreddits is None:
            subreddits = [
                'investing', 'stocks', 'SecurityAnalysis',
                'options', 'algotrading', 'StockMarket',
                'economics', 'finance', 'fatFIRE'
            ]
        
        all_posts = []
        for sub in subreddits:
            try:
                subreddit = self.reddit.subreddit(sub)
                posts = subreddit.search(self.query, limit=limit, sort='top', time_filter='month')
                
                for post in posts:
                    # Filter quality
                    if post.score > 10 and post.num_comments > 5:
                        all_posts.append({
                            'title': post.title,
                            'content': post.selftext,
                            'score': post.score,
                            'comments': post.num_comments,
                            'url': post.url,
                            'subreddit': sub,
                            'date': datetime.fromtimestamp(post.created_utc),
                            'source': f'Reddit r/{sub}'
                        })
            except:
                continue
        
        return pd.DataFrame(all_posts)
    
class UnifiedFinancialScraper:
    def __init__(self, query):
        load_dotenv()
        self.client_secret = os.getenv("client_secret")
        self.client_id = os.getenv("client_id")
        self.user_agent = os.getenv("user_agent")
        self.config = config
        self.rss_feeds = RSSFeedScraper(query=query)
        self.alpha_vantage_scraper = AlphaVantageScraper(query=query)
        self.newsapi_scraper = NewsAPIScraper(query=query)
        self.reddit_scraper = RedditScraper(query=query)
        self.google_news_scraper = GoogleNewsScraper(query=query)
        self.finnhub_scraper = FinnhubScraper(query=query, q_type="general")
    
    def _deduplicate(self, df):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform(df['content'].fillna(''))
        similarity = cosine_similarity(tfidf)
        to_drop = set()
        for i in range(similarity.shape[0]):
            for j in range(i+1, similarity.shape[1]):
                if similarity[i, j] > 0.8:
                    to_drop.add(j)
        return df.drop(to_drop, axis=0).reset_index(drop=True)
    
    def _score_quality(self, df, source_column='source'):
        source_scores = {
            'Finnhub': 1,
            'Google News': 0.9,
            'RSS Feeds': 0.8,
            'Reddit': 0.4,
            'AlphaVantage': 0.85,
            'NewsAPI': 0.75
        }
        
        df['quality_score'] = df[source_column].map(source_scores).fillna(0.5)
        return df.sort_values(by='quality_score', ascending=False)
    
    def fetch_all(self, include_social=True):
        all_data = []
        
        # Fetch RSS Feed Data
        rss_data = self.rss_feeds.fetch_data(max_articles_per_feed=self.config.get("max_rss_articles", 5))
        all_data.append(rss_data)
        logger.info(f"Fetched {len(rss_data)} RSS articles")

        # Fetch Google News Data
        google_news_data = self.google_news_scraper.fetch_data(num_articles=self.config.get("max_google_news_articles", 5))
        all_data.append(google_news_data)
        logger.info(f"Fetched {len(google_news_data)} Google News articles")

        # Fetch Finnhub Market News
        finnhub_data = self.finnhub_scraper.fetch_data()
        all_data.append(finnhub_data)
        logger.info(f"Fetched {len(finnhub_data)} Finnhub articles")

        # Fetch Alpha Vantage News
        alpha_vantage_data = self.alpha_vantage_scraper.fetch_data()
        all_data.append(alpha_vantage_data)
        logger.info(f"Fetched {len(alpha_vantage_data)} Alpha Vantage articles")

        # Fetch NewsAPI Data
        newsapi_data = self.newsapi_scraper.fetch_data()
        all_data.append(newsapi_data)
        logger.info(f"Fetched {len(newsapi_data)} NewsAPI articles")

        if include_social:
            # Fetch Reddit Data
            reddit_data = self.reddit_scraper.fetch_data(limit=self.config.get("max_reddit_posts", 50))
            all_data.append(reddit_data)
            logger.info(f"Fetched {len(reddit_data)} Reddit posts")

        # Combine all data into a single DataFrame
        combined_data = pd.concat(all_data, ignore_index=True)
        logger.info(f"Total articles before deduplication: {len(combined_data)}")
        df = self._score_quality(combined_data)
        print(df[['title', 'source', 'quality_score']].head(10))
        df = self._deduplicate(df)
        logger.info(f"Total articles after deduplication: {len(df)}")
        print(df["source"].value_counts())
        return df

if __name__ == "__main__":

    scraper = UnifiedFinancialScraper(query="stock market")
    result_df = scraper.fetch_all(include_social=True)
    print(f"\nFetched {len(result_df)} articles")
    print(result_df[['title', 'source', 'quality_score']].head(10))