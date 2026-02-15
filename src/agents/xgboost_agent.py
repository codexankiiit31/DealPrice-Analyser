import os
import numpy as np
from sentence_transformers import SentenceTransformer
import joblib
from src.agents.agent import Agent

class XGBoostPriceAgent(Agent):

    name = "XGBPrice Agent"
    color = Agent.MAGENTA

    # Load the embedding model once and reuse
    _vectorizer = SentenceTransformer("intfloat/e5-small-v2")

    def __init__(self):
        """
        Initialize this object by loading in the saved model weights
        and the SentenceTransformer vector encoding model
        """
        self.log("XGBoost Agent is initializing")

        # Load model path relative to project root
        # PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        # MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "xgboost_model.pkl")
        self.model = joblib.load("models/xgboost_model.pkl")

        self.vectorizer = self._vectorizer
        self.log("XGBoost Agent is ready")

    def price(self, description: str) -> float:
        """
        Use a XGBoost model to estimate the price of the described item
        :param description: the product to be estimated
        :return: the price as a float
        """        
        self.log("XGBoost Agent is starting a prediction")
        vector = self.vectorizer.encode([description])
        result = max(0, self.model.predict(vector)[0])
        self.log(f"XGBoost Agent completed - predicting ${result:.2f}")
        return result