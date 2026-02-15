from pydantic import BaseModel
from typing import List, Dict, Self
from bs4 import BeautifulSoup
from src.config.feeds import CATEGORY_FEEDS
import re
import feedparser
from tqdm import tqdm
import requests
import time

def extract(html_snippet: str) -> str:
    """
    Use Beautiful Soup to clean up this HTML snippet and extract useful text
    """
    soup = BeautifulSoup(html_snippet, 'html.parser')
    snippet_div = soup.find('div', class_='snippet summary')
    
    if snippet_div:
        description = snippet_div.get_text(strip=True)
        description = BeautifulSoup(description, 'html.parser').get_text()
        description = re.sub('<[^<]+?>', '', description)
        result = description.strip()
    else:
        result = html_snippet
    return result.replace('\n', ' ')

class ScrapedDeal:
    """
    A class to represent a Deal retrieved from an RSS feed
    """
    category: str
    title: str
    summary: str
    url: str
    details: str
    features: str

    def __init__(self, entry: Dict[str, str]):
        """
        Populate this instance based on the provided dict
        """
        self.title = entry['title']
        self.summary = extract(entry['summary'])
        self.url = entry['links'][0]['href']
        stuff = requests.get(self.url).content
        soup = BeautifulSoup(stuff, 'html.parser')
        content = soup.find('div', class_='content-section').get_text()
        content = content.replace('\nmore', '').replace('\n', ' ')
        if "Features" in content:
            self.details, self.features = content.split("Features")
        else:
            self.details = content
            self.features = ""

    def __repr__(self):
        """
        Return a string to describe this deal
        """
        return f"<{self.title}>"

    def describe(self):
        """
        Return a longer string to describe this deal for use in calling a model
        """
        return f"Title: {self.title}\nDetails: {self.details.strip()}\nFeatures: {self.features.strip()}\nURL: {self.url}"

    @classmethod
    def fetch(cls, show_progress: bool = False, selected_feeds: List[str] = None) -> List[Self]:
        """
        Retrieve all deals from the selected RSS feeds.
        If selected_feeds is None, fall back to all feeds in CATEGORY_FEEDS.
        """
        deals = []
        feed_list = selected_feeds if selected_feeds else [list(CATEGORY_FEEDS.values())[0]]
        feed_iter = tqdm(feed_list) if show_progress else feed_list

        for feed_url in feed_iter:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:5]:
                deals.append(cls(entry))
                time.sleep(0.5)
        return deals

class Deal(BaseModel):
    """
    A class to Represent a Deal with a summary description
    """
    product_description: str
    price: float
    url: str

class DealSelection(BaseModel):
    """
    A class to Represent a list of Deals
    """
    deals: List[Deal]

class Opportunity(BaseModel):
    """
    A class to represent a possible opportunity: a Deal where we estimate
    it should cost more than it's being offered
    """
    deal: Deal
    estimate: float
    discount: float