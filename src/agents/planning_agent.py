from typing import Optional, List
from src.agents.agent import Agent
from src.agents.deals import ScrapedDeal, DealSelection, Deal, Opportunity
from src.agents.scanner_agent import ScannerAgent
from src.agents.ensemble_agent import EnsembleAgent


class PlanningAgent(Agent):

    name = "Planning Agent"
    color = Agent.GREEN
    DEAL_THRESHOLD = 50

    def __init__(self, collection):
        """
        Create instances of the 3 Agents that this planner coordinates across
        """
        self.log("Planning Agent is initializing")
        self.scanner = ScannerAgent()
        self.ensemble = EnsembleAgent(collection)
        self.log("Planning Agent is ready")

    def run(self, deal: Deal) -> Opportunity:
        """
        Run the workflow for a particular deal
        :param deal: the deal, summarized from an RSS scrape
        :returns: an opportunity including the discount
        """
        self.log("Planning Agent is pricing up a potential deal")
        estimate = self.ensemble.price(deal.product_description)
        discount = estimate - deal.price
        self.log(f"Planning Agent has processed a deal with discount ${discount:.2f}")
        return Opportunity(deal=deal, estimate=estimate, discount=discount)

    def plan(self, memory: List[str] = [], selected_feeds=None) -> Optional[List[Opportunity]]:
        """
        Run the full workflow:
        1. Use the ScannerAgent to find deals from RSS feeds
        2. Use the EnsembleAgent to estimate them
        :param memory: a list of URLs that have been surfaced in the past
        :param selected_feeds: a list of feeds to scan
        :return: a list of up to 3 Opportunities above threshold, or None if no deals pass threshold
        """
        self.log("Planning Agent is kicking off a run with selected feeds")
        selection = self.scanner.scan(memory=memory, selected_feeds=selected_feeds)

        if selection:
            # Price each deal and sort them by discount in descending order
            opportunities = [self.run(deal) for deal in selection.deals[:5]]
            opportunities.sort(key=lambda opp: opp.discount, reverse=True)

            # Filter deals above threshold and take up to 3
            above_threshold = [opp for opp in opportunities if opp.discount > self.DEAL_THRESHOLD]

            if above_threshold:
                deals_to_return = above_threshold[:3]
                self.log(f"Planning Agent found {len(deals_to_return)} deals above threshold")
                for i, deal in enumerate(deals_to_return, 1):
                    self.log(f"Planning Agent identified Deal {i} with discount ${deal.discount:.2f}")
                self.log("Planning Agent has completed a run")
                return deals_to_return
            else:
                self.log("No deals passed the discount threshold")
                return None

        return None