import os
import sys
import logging
import json
from typing import List, Optional
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from src.agents.planning_agent import PlanningAgent
from src.agents.deals import Opportunity
from src.config.feeds import CATEGORY_FEEDS
from sklearn.manifold import TSNE
import numpy as np


# Colors for logging
BG_BLUE = '\033[44m'
WHITE = '\033[37m'
RESET = '\033[0m'

# Colors for plot
CATEGORIES = ['Appliances', 'Automotive', 'Cell_Phones_and_Accessories', 'Electronics','Musical_Instruments', 'Office_Products', 'Tools_and_Home_Improvement', 'Toys_and_Games']
COLORS = ['red', 'blue', 'brown', 'orange', 'yellow', 'green' , 'purple', 'cyan']

def init_logging():
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] [Agents] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %z",
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)


class DealAgentFramework:

    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DB = str(PROJECT_ROOT / "chroma_db")
    MEMORY_FILENAME = str(PROJECT_ROOT / "data" / "memory.json")


    def __init__(self):
        init_logging()
        load_dotenv()
        client = chromadb.PersistentClient(path=self.DB)
        self.memory = self.read_memory()
        self.collection = client.get_or_create_collection("price_items")
        self.planner = None

    def init_agents_as_needed(self):
        if not self.planner:
            self.log("Initializing Agent Framework")
            self.planner = PlanningAgent(self.collection)
            self.log("Agent Framework is ready")

        
    def read_memory(self) -> List[Opportunity]:
        if os.path.exists(self.MEMORY_FILENAME):
            with open(self.MEMORY_FILENAME, "r") as file:
                data = json.load(file)
            try:
                opportunities = [Opportunity(**item) for item in data]
            except Exception as e:
                logging.error(f"Error loading memory.json: {e}")
                opportunities = []
            return opportunities
        return []


    def write_memory(self) -> None:
        data = [opportunity.dict() for opportunity in self.memory]
        with open(self.MEMORY_FILENAME, "w") as file:
            json.dump(data, file, indent=2)


    def log(self, message: str):
        text = BG_BLUE + WHITE + "[Agent Framework] " + message + RESET
        logging.info(text)


    def run(self, selected_categories: List[str] = None) -> List[Opportunity]:
        """
        Run the Planning Agent workflow:
        1. Initialize sub-agents if needed
        2. Map selected categories to their corresponding feeds
        3. Execute the planner to surface opportunities
        4. Handle both single and multiple returned opportunities
        :param selected_categories: list of categories chosen by the user
        :return: a list of Opportunities surfaced in the current session
        """
        self.init_agents_as_needed()
        self.log("Planning Agent is kicking off a run")

        # Map selected categories to feeds
        selected_feeds = None
        if selected_categories:
            selected_feeds = [
                CATEGORY_FEEDS[cat] for cat in selected_categories if cat in CATEGORY_FEEDS
            ]

        # Run the planner with current memory and feeds
        result = self.planner.plan(memory=self.memory, selected_feeds=selected_feeds)
        self.log(f"Planning Agent has completed and returned: {result}")

        current_session_deals = []

        if result:
            if isinstance(result, list):
                # Multiple deals returned
                for deal in result:
                    self.memory.append(deal)
                    current_session_deals.append(deal)
                self.log(f"Planning Agent found {len(result)} deals above threshold in current session")
            else:
                # Backward compatibility: single deal returned
                self.memory.append(result)
                current_session_deals.append(result)
                self.log("Planning Agent found 1 deal above threshold in current session")

            self.write_memory()
        else:
            self.log("Planning Agent found no deals above threshold in current session")

        return current_session_deals


    @classmethod
    def get_plot_data(cls, max_datapoints=10000):
        client = chromadb.PersistentClient(path=cls.DB)
        collection = client.get_or_create_collection("price_items")
        result = collection.get(include=['embeddings', 'documents', 'metadatas'], limit=max_datapoints)
        vectors = np.array(result['embeddings'])
        documents = result['documents']
        colors = ['gray'] * len(documents)
        tsne = TSNE(n_components=3, random_state=42)
        reduced_vectors = tsne.fit_transform(vectors)
        return documents, reduced_vectors, colors



if __name__=="__main__":
    DealAgentFramework().run()