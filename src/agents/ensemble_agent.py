import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

from src.agents.agent import Agent
from src.agents.specialist_agent import SpecialistAgent
from src.agents.frontier_agent import FrontierAgent
from src.agents.xgboost_agent import XGBoostPriceAgent


class EnsembleAgent(Agent):

    name = "Ensemble Agent"
    color = Agent.YELLOW

    
    def __init__(self, collection):
        """
        Create an instance of Ensemble, by creating each of the models
        And loading the weights of the Ensemble
        """
        self.log("Initializing Ensemble Agent")
        self.specialist = SpecialistAgent()
        self.frontier = FrontierAgent(collection)
        self.xgboost = XGBoostPriceAgent()

        ## PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        ## MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "ensemble_model.pkl")
        self.model = joblib.load("models/ensemble_model.pkl")

        self.log("Ensemble Agent is ready")


    def price(self, description: str) -> float:
        """
        Run this ensemble model
        Ask each of the models to price the product
        Then use the Linear Regression model to return the weighted price
        """
        self.log("Running Ensemble Agent - collaborating with specialist, frontier and xgboost agents")
        
        specialist = self.frontier.price(description)
        frontier = self.frontier.price(description)
        xgboost = self.xgboost.price(description)

        # Compute aggregate features
        max_value = max(specialist, frontier, xgboost)
        mean_value = np.mean([specialist, frontier, xgboost])

        X = pd.DataFrame({
            "FT_LLaMA": [specialist],
            "GPT4oMini": [frontier],
            "XGBoost": [xgboost],
            "Max": [max_value],
            "Mean": [mean_value]
        })

        y = max(0, self.model.predict(X)[0])
        self.log(f"Ensemble Agent complete - returning ${y:.2f}")
        return y
