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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

class BaseScraper:
    def __init__(self, query, q_type=None, from_date=None, to_date=None): # q_type could be 'ticker: company_news' or 'general : market_news'
        self.query = query
        self.q_type = q_type
        self.from_date = from_date
        self.to_date = to_date

    def fetch_data(self):
        raise NotImplementedError("This method should be overridden by subclasses.")

class FinnhubScraper(BaseScraper):
    def __init__(self, api_key, query, q_type, from_date=None, to_date=None):
        super().__init__(query, q_type, from_date, to_date)
        self.api_key = api_key
        self.base_url = "https://finnhub.io/api/v1/"

    def _fetch_company_news(self, num_articles):
        url = f"{self.base_url}company-news"
        params = {
            "symbol": self.query,
            "from": self.from_date,
            "to": self.to_date,
            "token": self.api_key
        }
        response = requests.get(url, params=params).json()
        articles = [
            {
                "title": a["headline"],
                "content": a.get("summary", ""),
                "source": a["source"],
                "date": datetime.fromtimestamp(a["datetime"]).isoformat()
            }
            for a in response
        ][:num_articles]
        df = pd.DataFrame(articles)
        return df

    def _fetch_market_news(self, num_articles):
        url = f"{self.base_url}news"
        params = {
            "category": "general",
            "token": self.api_key,
            "from": self.from_date,
            "to": self.to_date
        }
        response = requests.get(url, params=params).json()
        articles = [
            {
                "title": a["headline"],
                "content": a.get("summary", ""),
                "source": a["source"],
                "date": datetime.fromtimestamp(a["datetime"]).isoformat()
            }
            for a in articles if self.query.lower() in a["headline"].lower()
        ][:num_articles]
        df = pd.DataFrame(articles)
        return df
    
    def fetch_data(self):
        if self.q_type == "ticker":
            return self._fetch_company_news(num_articles=50)
        elif self.q_type == "general":
            return self._fetch_market_news(num_articles=50)
        else:
            raise ValueError("q_type must be either 'ticker' or 'general'.")

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
    def fetch_data(self, max_articles_per_feed=5):
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
    
class TwitterScraper(BaseScraper):
    def __init__(self, query, mode):
        super().__init__(query)
        self.scraper = ntscraper.Nitter()
        self.mode = mode  # 'search' or 'user'
    
    def _accounts_search(self, accounts: list = None):
        all_tweets = []
        if accounts is None:
            accounts = [
                'business', 'FT', 'WSJ', 'Bloomberg', 'economics',
                'CNBC', 'MarketWatch', 'Reuters', 'YahooFinance', 'Forbes'
            ]
        elif isinstance(accounts, str):
            accounts = [accounts]

        for account in accounts:
            try:
                tweets = self.scraper.get_tweets(account, mode="user", number=100)
                for tweet in tweets["tweets"]:
                    all_tweets.append({
                    'title': tweet.get('text', '')[:100],  # First 100 chars
                    'content': tweet.get('text', ''),
                    'source': f"Twitter @{tweet.get('user', {}).get('username', 'unknown')}",
                    'date': tweet.get('date'),
                    'url': tweet.get('link', ''),
                    'engagement': tweet.get('stats', {}).get('likes', 0)
                    })
            except Exception as e:
                logger.error(f"Error fetching tweets from {account}: {e}")
                continue
        df = pd.DataFrame(all_tweets)
        return df
    def _query_search(self, query):
        all_tweets = []
        try:
            tweets = self.scraper.get_tweets(query, mode="search", number=100)
            for tweet in tweets["tweets"]:
                all_tweets.append({
                    'title': tweet.get('text', '')[:100],  # First 100 chars
                    'content': tweet.get('text', ''),
                    'source': f"Twitter @{tweet.get('user', {}).get('username', 'unknown')}",
                    'date': tweet.get('date'),
                    'url': tweet.get('link', ''),
                    'engagement': tweet.get('stats', {}).get('likes', 0)
                })
        except Exception as e:
            logger.error(f"Error fetching tweets for query '{query}': {e}")
        
        df = pd.DataFrame(all_tweets)
        return df
    def fetch_data(self, accounts=None):
        if self.mode == "user":
            return self._accounts_search(accounts)
        elif self.mode == "search":
            return self._query_search(self.query)
        else:
            raise ValueError("mode must be either 'user' or 'search'.")

class GoogleNewsScraper(BaseScraper):
    def __init__(self, query):
        super().__init__(query)
        self.config = Config()
        self.config.request_timeout = 10
        self.config.browser_user_agent = "Mozilla/5.0"

    def fetch_data(self, num_articles=100):
        url = f"https://news.google.com/rss/search?q={self.query}&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(url)
        all_articles = []
        for entry in feed.entries[:num_articles]:
            try:
                article = Article(entry.link, config=self.config)
                article.download()
                article.parse()
                all_articles.append({
                    "title": article.title,
                    "content": article.text,
                    "source": "Google News",
                    "date": datetime(*entry.published_parsed[:6]) if hasattr(entry, "published_parsed") else None,
                    "url": entry.link
                })
            except Exception as e:
                logger.error(f"Error processing article {entry.link}: {e}")
                continue

        df = pd.DataFrame(all_articles)
        return df

class RedditScraper(BaseScraper):
    def __init__(self, query, client_id, client_secret, user_agent):
        super().__init__(query)
        self.reddit = praw.Reddit(client_id=client_id,
                                  client_secret=client_secret,
                                  user_agent=user_agent)

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
    def __init__(self, query,config):
        load_dotenv()
        self.client_secret = os.getenv("client_secret")
        self.client_id = os.getenv("client_id")
        self.user_agent = os.getenv("user_agent")
        self.finnhub_api_key = os.getenv("FINHUB_API_KEY")
        self.config = config
        self.rss_feeds = RSSFeedScraper(query=query)
        self.reddit_scraper = RedditScraper(
            query=query,
            client_id=self.client_id,
            client_secret=self.client_secret,
            user_agent=self.user_agent
        )
        self.google_news_scraper = GoogleNewsScraper(query=query)
        self.finnhub_scraper = FinnhubScraper(
            api_key=self.finnhub_api_key,
            query=query,
            q_type="general"
        )
        self.twitter_scraper = TwitterScraper(query=query, mode="search")
    
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
            'Twitter': 0.6
        }
        df['quality_score'] = df[source_column].map(source_scores).fillna(0.5)
        return df.sort_values(by='quality_score', ascending=False)
    
    def fetch_all(self, include_social=True):
        all_data = []
        
        # Fetch RSS Feed Data
        rss_data = self.rss_feeds.fetch_data(max_articles_per_feed=self.config.get("max_rss_articles", 5))
        all_data.append(rss_data)
        
        # Fetch Google News Data
        google_news_data = self.google_news_scraper.fetch_data(num_articles=self.config.get("max_google_news_articles", 5))
        all_data.append(google_news_data)
        
        # Fetch Finnhub Market News
        finnhub_data = self.finnhub_scraper.fetch_data()
        all_data.append(finnhub_data)
        
        if include_social:
            # Fetch Reddit Data
            reddit_data = self.reddit_scraper.fetch_data(limit=self.config.get("max_reddit_posts", 50))
            all_data.append(reddit_data)
            
            # Fetch Twitter Data
            twitter_data = self.twitter_scraper.fetch_data()
            all_data.append(twitter_data)
        
        # Combine all data into a single DataFrame
        combined_data = pd.concat(all_data, ignore_index=True)
        df = self._deduplicate(combined_data)
        df = self._score_quality(df)
        logger.info(f"Total articles after deduplication: {len(df)}")
        return df