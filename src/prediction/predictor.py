"""
Match Predictor
Makes predictions for upcoming matches
"""
import pandas as pd
import numpy as np
from datetime import datetime
import config


class MatchPredictor:
    """Predicts outcomes for football matches"""
    
    def __init__(self, model, feature_engineer):
        """
        Args:
            model: Trained prediction model
            feature_engineer: FeatureEngineer instance for creating features
        """
        self.model = model
        self.feature_engineer = feature_engineer
        
        if not model.is_trained:
            raise ValueError("Model must be trained before making predictions")
    
    def predict_match(self, match_data, historical_data):
        """
        Predict outcome for a single match
        
        Args:
            match_data: Dict with match info (home_team, away_team, date)
            historical_data: DataFrame with historical matches for feature creation
        
        Returns:
            Dictionary with predictions
        """
        # Ensure datetime consistency (remove timezone for compatibility)
        historical_data = historical_data.copy()
        if 'date' in historical_data.columns:
            historical_data['date'] = pd.to_datetime(historical_data['date']).dt.tz_localize(None)
        
        # Calculate features directly from historical data for these two teams
        home_team = match_data["home_team"]
        away_team = match_data["away_team"]
        match_date = pd.to_datetime(match_data.get("date", datetime.now())).tz_localize(None)
        
        # Get historical matches before this date
        hist_before = historical_data[historical_data['date'] < match_date].copy()
        
        if len(hist_before) < 10:
            return {
                "home_team": home_team,
                "away_team": away_team,
                "error": "Insufficient historical data (need at least 10 matches)"
            }
        
        # Create a row with the match and extract features manually
        # Get form for both teams
        home_recent = hist_before[
            (hist_before['home_team'] == home_team) | 
            (hist_before['away_team'] == home_team)
        ].tail(5)
        
        away_recent = hist_before[
            (hist_before['home_team'] == away_team) | 
            (hist_before['away_team'] == away_team)
        ].tail(5)
        
        if len(home_recent) < 5 or len(away_recent) < 5:
            return {
                "home_team": home_team,
                "away_team": away_team,
                "error": "Insufficient recent matches for form calculation"
            }
        
        # Build a minimal DataFrame with just this match for feature engineering
        match_df = pd.DataFrame([{
            'date': match_date,
            'home_team': home_team,
            'away_team': away_team,
            'home_score': 0,  # Dummy
            'away_score': 0,  # Dummy
            'result': 'H',  # Dummy
            'competition': match_data.get('competition', 'Unknown')
        }])
        
        # Combine with ALL historical data, then feature engineer
        combined_df = pd.concat([hist_before, match_df], ignore_index=True)
        
        # Create features
        featured_df = self.feature_engineer.create_all_features(combined_df)
        
        # Find our prediction match in the featured dataframe
        # It should be one of the last rows that matches our team names
        prediction_row = featured_df[
            (featured_df['home_team'] == match_data['home_team']) & 
            (featured_df['away_team'] == match_data['away_team'])
        ].tail(1)
        
        # Get the last row (our match)
        if len(prediction_row) == 0:
            return {
                "home_team": match_data["home_team"],
                "away_team": match_data["away_team"],
                "error": "Insufficient historical data to make prediction"
            }
        
        match_features = prediction_row[self.model.feature_columns]
        
        # Make prediction
        probabilities = self.model.predict_proba(match_features)[0]
        
        prediction = {
            "home_team": match_data["home_team"],
            "away_team": match_data["away_team"],
            "date": match_data.get("date", "Unknown"),
            "probabilities": {
                "home_win": float(probabilities[0]),
                "draw": float(probabilities[1]),
                "away_win": float(probabilities[2])
            },
            "predicted_outcome": self._get_predicted_outcome(probabilities),
            "confidence": float(np.max(probabilities))
        }
        
        return prediction
    
    def predict_matches(self, matches_df, historical_data):
        """
        Predict outcomes for multiple matches
        
        Args:
            matches_df: DataFrame with upcoming matches
            historical_data: DataFrame with historical matches
        
        Returns:
            List of prediction dictionaries
        """
        predictions = []
        
        for idx, match in matches_df.iterrows():
            match_data = {
                "home_team": match["home_team"],
                "away_team": match["away_team"],
                "date": match.get("date", datetime.now()),
                "home_team_id": match.get("home_team_id"),
                "away_team_id": match.get("away_team_id"),
                "competition": match.get("competition", "Unknown"),
                "competition_code": match.get("competition_code", ""),
                # Dummy values for feature engineering (won't be used)
                "home_score": 0,
                "away_score": 0,
                "result": "H"
            }
            
            prediction = self.predict_match(match_data, historical_data)
            
            # Add match ID if available
            if "match_id" in match:
                prediction["match_id"] = match["match_id"]
            
            predictions.append(prediction)
            
            # Display prediction or error
            print(f"Predicted: {prediction['home_team']} vs {prediction['away_team']}")
            if 'error' in prediction:
                print(f"  ⚠️  {prediction['error']}\n")
            else:
                print(f"  {prediction['predicted_outcome']} (confidence: {prediction['confidence']:.2%})")
                print(f"  Probabilities: H={prediction['probabilities']['home_win']:.2%}, "
                      f"D={prediction['probabilities']['draw']:.2%}, "
                      f"A={prediction['probabilities']['away_win']:.2%}\n")
        
        return predictions
    
    def _get_predicted_outcome(self, probabilities):
        """Convert probabilities to readable outcome"""
        outcomes = ["Home Win", "Draw", "Away Win"]
        return outcomes[np.argmax(probabilities)]
    
    def filter_high_confidence(self, predictions, min_confidence=None):
        """
        Filter predictions by confidence threshold
        
        Args:
            predictions: List of prediction dictionaries
            min_confidence: Minimum confidence threshold (uses config default if None)
        
        Returns:
            Filtered list of predictions
        """
        threshold = min_confidence or config.MIN_CONFIDENCE
        
        filtered = [
            pred for pred in predictions
            if 'confidence' in pred and pred["confidence"] >= threshold
        ]
        
        print(f"High confidence predictions ({threshold:.0%}+): {len(filtered)}/{len(predictions)}")
        return filtered
    
    def to_dataframe(self, predictions):
        """Convert predictions to DataFrame for easy viewing"""
        rows = []
        
        for pred in predictions:
            row = {
                "home_team": pred["home_team"],
                "away_team": pred["away_team"],
                "date": pred.get("date", ""),
                "predicted_outcome": pred["predicted_outcome"],
                "confidence": pred["confidence"],
                "home_win_prob": pred["probabilities"]["home_win"],
                "draw_prob": pred["probabilities"]["draw"],
                "away_win_prob": pred["probabilities"]["away_win"]
            }
            rows.append(row)
        
        return pd.DataFrame(rows)
