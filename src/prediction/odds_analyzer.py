"""
Odds Analyzer
Compares model predictions with bookmaker odds to identify value bets
"""
import pandas as pd
import numpy as np
import config


class OddsAnalyzer:
    """Analyzes betting odds and identifies value bets"""
    
    def __init__(self, value_threshold=None):
        """
        Args:
            value_threshold: Minimum edge over bookmaker (default from config)
        """
        self.value_threshold = value_threshold or config.VALUE_BET_THRESHOLD
    
    def odds_to_probability(self, odds):
        """
        Convert decimal odds to implied probability
        
        Args:
            odds: Decimal odds (e.g., 2.50)
        
        Returns:
            Implied probability (0-1)
        """
        return 1 / odds if odds > 0 else 0
    
    def probability_to_odds(self, probability):
        """
        Convert probability to decimal odds
        
        Args:
            probability: Probability (0-1)
        
        Returns:
            Decimal odds
        """
        return 1 / probability if probability > 0 else 0
    
    def calculate_expected_value(self, model_prob, bookmaker_odds):
        """
        Calculate expected value of a bet
        
        EV = (probability × (odds - 1)) - (1 - probability)
        
        Args:
            model_prob: Model's probability estimate (0-1)
            bookmaker_odds: Bookmaker's decimal odds
        
        Returns:
            Expected value (positive = profitable bet)
        """
        return (model_prob * (bookmaker_odds - 1)) - (1 - model_prob)
    
    def calculate_kelly_stake(self, model_prob, bookmaker_odds, bankroll=100, kelly_fraction=0.25):
        """
        Calculate optimal stake using Kelly Criterion
        
        Args:
            model_prob: Model's probability estimate (0-1)
            bookmaker_odds: Bookmaker's decimal odds
            bankroll: Total bankroll
            kelly_fraction: Fraction of Kelly to use (0.25 = quarter Kelly, conservative)
        
        Returns:
            Recommended stake amount
        """
        bookmaker_prob = self.odds_to_probability(bookmaker_odds)
        
        # Kelly formula: f = (bp - q) / b
        # where b = odds - 1, p = model probability, q = 1 - p
        b = bookmaker_odds - 1
        p = model_prob
        q = 1 - p
        
        kelly = (b * p - q) / b
        
        # Apply fractional Kelly for safety
        kelly = kelly * kelly_fraction
        
        # Ensure non-negative
        kelly = max(0, kelly)
        
        # Calculate stake
        stake = bankroll * kelly
        
        return stake
    
    def find_value_bets(self, predictions, odds_data):
        """
        Identify value bets by comparing predictions with bookmaker odds
        
        Args:
            predictions: List of prediction dictionaries from MatchPredictor
            odds_data: DataFrame or dict with bookmaker odds
                      Format: {match_id: {"home": odds, "draw": odds, "away": odds}}
        
        Returns:
            List of value bet opportunities
        """
        value_bets = []
        
        for pred in predictions:
            # Get match identifier
            match_key = f"{pred['home_team']} vs {pred['away_team']}"
            
            # Skip if no odds data available
            if match_key not in odds_data and "match_id" not in pred:
                continue
            
            # Get odds
            if "match_id" in pred and pred["match_id"] in odds_data:
                odds = odds_data[pred["match_id"]]
            elif match_key in odds_data:
                odds = odds_data[match_key]
            else:
                continue
            
            # Analyze each outcome
            outcomes = [
                ("home_win", "home", pred["probabilities"]["home_win"]),
                ("draw", "draw", pred["probabilities"]["draw"]),
                ("away_win", "away", pred["probabilities"]["away_win"])
            ]
            
            for outcome_name, odds_key, model_prob in outcomes:
                if odds_key not in odds:
                    continue
                
                bookmaker_odds = odds[odds_key]
                bookmaker_prob = self.odds_to_probability(bookmaker_odds)
                
                # Calculate edge (model probability - bookmaker probability)
                edge = model_prob - bookmaker_prob
                
                # Calculate expected value
                ev = self.calculate_expected_value(model_prob, bookmaker_odds)
                
                # Check if it's a value bet
                if edge >= self.value_threshold and ev > 0:
                    kelly_stake = self.calculate_kelly_stake(model_prob, bookmaker_odds)
                    
                    value_bet = {
                        "match": match_key,
                        "home_team": pred["home_team"],
                        "away_team": pred["away_team"],
                        "date": pred.get("date", ""),
                        "outcome": outcome_name.replace("_", " ").title(),
                        "model_probability": model_prob,
                        "bookmaker_odds": bookmaker_odds,
                        "bookmaker_probability": bookmaker_prob,
                        "edge": edge,
                        "expected_value": ev,
                        "kelly_stake_pct": kelly_stake,  # Percentage of bankroll
                        "confidence": pred["confidence"]
                    }
                    
                    value_bets.append(value_bet)
        
        # Sort by expected value (best opportunities first)
        value_bets.sort(key=lambda x: x["expected_value"], reverse=True)
        
        return value_bets
    
    def display_value_bets(self, value_bets, top_n=10):
        """Display value bets in a readable format"""
        if not value_bets:
            print("No value bets found")
            return
        
        print(f"\n{'='*80}")
        print(f"VALUE BET OPPORTUNITIES (Top {min(top_n, len(value_bets))})")
        print(f"{'='*80}\n")
        
        for i, bet in enumerate(value_bets[:top_n], 1):
            print(f"{i}. {bet['match']}")
            print(f"   Date: {bet['date']}")
            print(f"   Bet: {bet['outcome']}")
            print(f"   Bookmaker Odds: {bet['bookmaker_odds']:.2f} (implied prob: {bet['bookmaker_probability']:.2%})")
            print(f"   Model Probability: {bet['model_probability']:.2%}")
            print(f"   Edge: {bet['edge']:.2%}")
            print(f"   Expected Value: {bet['expected_value']:.4f}")
            print(f"   Recommended Stake: {bet['kelly_stake_pct']:.1f}% of bankroll")
            print(f"   Confidence: {bet['confidence']:.2%}")
            print()
    
    def to_dataframe(self, value_bets):
        """Convert value bets to DataFrame"""
        return pd.DataFrame(value_bets)


# Example odds data structure for testing
EXAMPLE_ODDS = {
    "Example Match": {
        "home": 2.10,  # Home win odds
        "draw": 3.40,  # Draw odds
        "away": 3.50   # Away win odds
    }
}


if __name__ == "__main__":
    # Example usage
    analyzer = OddsAnalyzer()
    
    # Example prediction
    example_prediction = {
        "home_team": "Team A",
        "away_team": "Team B",
        "date": "2024-01-15",
        "probabilities": {
            "home_win": 0.55,
            "draw": 0.25,
            "away_win": 0.20
        },
        "predicted_outcome": "Home Win",
        "confidence": 0.55
    }
    
    # Example odds
    example_odds = {
        "Team A vs Team B": {
            "home": 2.10,  # Implied prob: 47.6%
            "draw": 3.40,  # Implied prob: 29.4%
            "away": 3.50   # Implied prob: 28.6%
        }
    }
    
    # Find value bets
    value_bets = analyzer.find_value_bets([example_prediction], example_odds)
    
    # Display results
    analyzer.display_value_bets(value_bets)
