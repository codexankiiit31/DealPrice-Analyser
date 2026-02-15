import os
import re
from typing import List

import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import chromadb

from src.agents.agent import Agent


class FrontierAgent(Agent):

    name = "Frontier Agent"
    color = Agent.BLUE

    MODEL = "gemini-2.5-flash-lite"

    # Shared encoder to avoid reloading overhead
    _encoder = SentenceTransformer("intfloat/e5-small-v2")

    def __init__(self, collection):
        """
        Set up this instance by connecting to Gemini,
        the Chroma Datastore, and the vector encoder
        """
        self.log("Initializing Frontier Agent")

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("❌ GEMINI_API_KEY environment variable is not set.")

        genai.configure(api_key=api_key)

        self.model = genai.GenerativeModel(
            model_name=self.MODEL,
            system_instruction="You estimate prices of items. Reply only with the price, no explanation."
        )

        self.collection = collection
        self.encoder = self._encoder

        self.log("Frontier Agent is ready (Gemini)")

    def make_context(self, similars: List[str], prices: List[float]) -> str:
        """
        Create context that can be inserted into the prompt
        """
        message = (
            "To provide some context, here are some other items that might be similar "
            "to the item you need to estimate.\n\n"
        )
        for similar, price in zip(similars, prices):
            message += (
                f"Potentially related product:\n{similar}\n"
                f"Price is ${price:.2f}\n\n"
            )
        return message

    def find_similars(self, description: str):
        """
        Return a list of items similar to the given one by looking in the Chroma datastore
        """
        self.log("Frontier Agent is performing a RAG search to find 5 similar products")

        vector = self.encoder.encode([description])
        results = self.collection.query(
            query_embeddings=vector.astype(float).tolist(),
            n_results=5
        )

        documents = results["documents"][0]
        # prices = [m["price"] for m in results["metadatas"][0]]
        prices = [m.get("price", m.get("selling_price", 0.0)) for m in results["metadatas"][0]]


        self.log("Frontier Agent has found similar products")
        return documents, prices

    def get_price(self, text: str) -> float:
        """
        Extract a floating point number from model output
        """
        text = text.replace("$", "").replace(",", "")
        match = re.search(r"[-+]?\d*\.\d+|\d+", text)
        return float(match.group()) if match else 0.0

    def price(self, description: str) -> float:
        """
        Estimate the price of the described product using Gemini + RAG
        """
        documents, prices = self.find_similars(description)

        prompt = self.make_context(documents, prices)
        prompt += "How much does this cost?\n\n"
        prompt += description + "\n\nPrice is $"

        self.log("Frontier Agent is calling Gemini")

        response = self.model.generate_content(
            prompt,
            generation_config={
                "temperature": 0,
                "max_output_tokens": 5
            }
        )

        reply = response.text.strip()
        result = self.get_price(reply)

        self.log(f"Frontier Agent completed - predicting ${result:.2f}")
        return result
