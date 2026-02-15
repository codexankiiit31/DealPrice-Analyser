import os
import json
from typing import Optional, List

import google.generativeai as genai
from dotenv import load_dotenv

from src.agents.deals import ScrapedDeal, DealSelection
from src.agents.agent import Agent

load_dotenv()


class ScannerAgent(Agent):

    MODEL = "gemini-2.5-flash-lite"

    SYSTEM_PROMPT = """You identify and summarize the 5 most detailed deals from a list, by selecting deals that have the most detailed, high quality description and the most clear price.
Respond strictly in JSON with no explanation, using this format. You should provide the price as a number derived from the description. If the price of a deal isn't clear, do not include that deal in your response.
Most important is that you respond with the 5 deals that have the most detailed product description with price. It's not important to mention the terms of the deal; most important is a thorough description of the product.
Be careful with products that are described as "$XXX off" or "reduced by $XXX" - this isn't the actual price of the product.

{"deals": [
  {
    "product_description": "A 4–5 sentence paragraph describing the product itself in detail.",
    "price": 99.99,
    "url": "the url as provided"
  }
]}"""

    USER_PROMPT_PREFIX = """Respond with the most promising 5 deals from this list, selecting those which have the most detailed, high quality product description and a clear price that is greater than 0.
Respond strictly in JSON, and only JSON. You should rephrase the description to be a summary of the product itself, not the terms of the deal.
Remember to respond with a paragraph of text in the product_description field for each of the 5 items that you select.
Be careful with products that are described as "$XXX off" or "reduced by $XXX" - this isn't the actual price of the product.

Deals:
"""

    USER_PROMPT_SUFFIX = "\n\nStrictly respond in JSON and include exactly 5 deals, no more."

    name = "Scanner Agent"
    color = Agent.CYAN

    def __init__(self):
        """
        Set up this instance by initializing Gemini
        """
        self.log("Scanner Agent is initializing")

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("❌ GEMINI_API_KEY environment variable is not set.")

        genai.configure(api_key=api_key)

        self.model = genai.GenerativeModel(
            model_name=self.MODEL,
            system_instruction=self.SYSTEM_PROMPT
        )

        self.log("Scanner Agent is ready (Gemini)")

    def fetch_deals(self, memory, selected_feeds=None) -> List[ScrapedDeal]:
        """
        Look up deals published on RSS feeds for the selected categories.
        Return any new deals that are not already in memory.
        """
        self.log("Scanner Agent is about to fetch deals from selected RSS feeds")
        urls = [opp.deal.url for opp in memory]
        scraped = ScrapedDeal.fetch(selected_feeds=selected_feeds)
        result = [scrape for scrape in scraped if scrape.url not in urls]
        self.log(f"Scanner Agent received {len(result)} new deals")
        return result

    def make_user_prompt(self, scraped) -> str:
        """
        Create a user prompt for Gemini based on the scraped deals provided
        """
        user_prompt = self.USER_PROMPT_PREFIX
        user_prompt += "\n\n".join([scrape.describe() for scrape in scraped])
        user_prompt += self.USER_PROMPT_SUFFIX
        return user_prompt

    def scan(
        self,
        memory: List[str] = [],
        selected_feeds=None
    ) -> Optional[DealSelection]:
        """
        Call Gemini to provide a high potential list of deals with good descriptions and prices
        :param memory: a list of URLs representing deals already raised
        :return: a selection of good deals, or None if there aren't any
        """
        scraped = self.fetch_deals(memory, selected_feeds)
        if not scraped:
            return None

        user_prompt = self.make_user_prompt(scraped)
        self.log("Scanner Agent is calling Gemini")

        try:
            response = self.model.generate_content(
                user_prompt,
                generation_config={
                    "temperature": 0,
                    "response_mime_type": "application/json"
                }
            )

            parsed = json.loads(response.text)
            result = DealSelection(**parsed)

        except Exception as e:
            self.log(f"❌ Scanner Agent encountered an error calling Gemini: {e}")
            return None

        result.deals = [deal for deal in result.deals if deal.price > 0]
        self.log(
            f"Scanner Agent received {len(result.deals)} selected deals with price>0 from Gemini"
        )
        return result
