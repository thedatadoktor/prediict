"""
Feature Engineering Module
Creates advanced features for match prediction
"""
import pandas as pd
import numpy as np
import config


class FeatureEngineer:
    """Creates features for football match prediction"""
    
    def __init__(self, form_matches=None, h2h_matches=None):
        self.form_matches = form_matches or config.FORM_MATCHES
        self.h2h_matches = h2h_matches or config.HEAD_TO_HEAD_MATCHES
    
    def create_all_features(self, df):
        """
        Create all features for match prediction
        
        Args:
            df: Match DataFrame
        
        Returns:
            DataFrame with features
        """
        df = df.copy()
        df = df.sort_values("date").reset_index(drop=True)
        
        print("Creating form features...")
        df = self._create_form_features(df)
        
        print("Creating head-to-head features...")
        df = self._create_h2h_features(df)
        
        print("Creating team strength features...")
        df = self._create_strength_features(df)
        
        print("Creating match context features...")
        df = self._create_context_features(df)
        
        # Remove rows with NaN (early matches without history)
        initial_rows = len(df)
        df = df.dropna()
        print(f"Removed {initial_rows - len(df)} matches without sufficient history")
        
        return df
    
    def _create_form_features(self, df):
        """Create recent form features"""
        df = df.copy()
        
        # Initialize feature columns
        for prefix in ["home", "away"]:
            df[f"{prefix}_form_points"] = np.nan
            df[f"{prefix}_form_goals_scored"] = np.nan
            df[f"{prefix}_form_goals_conceded"] = np.nan
            df[f"{prefix}_form_wins"] = np.nan
            df[f"{prefix}_form_draws"] = np.nan
            df[f"{prefix}_form_losses"] = np.nan
        
        # Calculate form for each match
        for idx in range(len(df)):
            match = df.iloc[idx]
            home_team = match["home_team"]
            away_team = match["away_team"]
            match_date = match["date"]
            
            # Get recent matches before this match
            recent_matches = df[df["date"] < match_date]
            
            # Home team form
            home_recent = recent_matches[
                (recent_matches["home_team"] == home_team) | 
                (recent_matches["away_team"] == home_team)
            ].tail(self.form_matches)
            
            if len(home_recent) >= self.form_matches:
                home_form = self._calculate_team_form(home_recent, home_team)
                for key, value in home_form.items():
                    df.at[idx, f"home_{key}"] = value
            
            # Away team form
            away_recent = recent_matches[
                (recent_matches["home_team"] == away_team) | 
                (recent_matches["away_team"] == away_team)
            ].tail(self.form_matches)
            
            if len(away_recent) >= self.form_matches:
                away_form = self._calculate_team_form(away_recent, away_team)
                for key, value in away_form.items():
                    df.at[idx, f"away_{key}"] = value
        
        return df
    
    def _calculate_team_form(self, matches, team):
        """Calculate form metrics for a team"""
        points = 0
        goals_scored = 0
        goals_conceded = 0
        wins = 0
        draws = 0
        losses = 0
        
        for _, match in matches.iterrows():
            is_home = match["home_team"] == team
            
            if is_home:
                goals_scored += match["home_score"]
                goals_conceded += match["away_score"]
                
                if match["result"] == "H":
                    points += 3
                    wins += 1
                elif match["result"] == "D":
                    points += 1
                    draws += 1
                else:
                    losses += 1
            else:
                goals_scored += match["away_score"]
                goals_conceded += match["home_score"]
                
                if match["result"] == "A":
                    points += 3
                    wins += 1
                elif match["result"] == "D":
                    points += 1
                    draws += 1
                else:
                    losses += 1
        
        return {
            "form_points": points,
            "form_goals_scored": goals_scored,
            "form_goals_conceded": goals_conceded,
            "form_wins": wins,
            "form_draws": draws,
            "form_losses": losses
        }
    
    def _create_h2h_features(self, df):
        """Create head-to-head features"""
        df = df.copy()
        
        # Initialize H2H columns
        df["h2h_home_wins"] = np.nan
        df["h2h_draws"] = np.nan
        df["h2h_away_wins"] = np.nan
        df["h2h_home_goals_avg"] = np.nan
        df["h2h_away_goals_avg"] = np.nan
        
        for idx in range(len(df)):
            match = df.iloc[idx]
            home_team = match["home_team"]
            away_team = match["away_team"]
            match_date = match["date"]
            
            # Get previous H2H matches
            h2h = df[
                (df["date"] < match_date) &
                (
                    ((df["home_team"] == home_team) & (df["away_team"] == away_team)) |
                    ((df["home_team"] == away_team) & (df["away_team"] == home_team))
                )
            ].tail(self.h2h_matches)
            
            if len(h2h) > 0:
                home_wins = 0
                away_wins = 0
                draws = 0
                home_goals = []
                away_goals = []
                
                for _, h2h_match in h2h.iterrows():
                    if h2h_match["home_team"] == home_team:
                        home_goals.append(h2h_match["home_score"])
                        away_goals.append(h2h_match["away_score"])
                        
                        if h2h_match["result"] == "H":
                            home_wins += 1
                        elif h2h_match["result"] == "D":
                            draws += 1
                        else:
                            away_wins += 1
                    else:
                        home_goals.append(h2h_match["away_score"])
                        away_goals.append(h2h_match["home_score"])
                        
                        if h2h_match["result"] == "A":
                            home_wins += 1
                        elif h2h_match["result"] == "D":
                            draws += 1
                        else:
                            away_wins += 1
                
                df.at[idx, "h2h_home_wins"] = home_wins
                df.at[idx, "h2h_draws"] = draws
                df.at[idx, "h2h_away_wins"] = away_wins
                df.at[idx, "h2h_home_goals_avg"] = np.mean(home_goals)
                df.at[idx, "h2h_away_goals_avg"] = np.mean(away_goals)
        
        return df
    
    def _create_strength_features(self, df):
        """Create team strength features based on historical performance"""
        df = df.copy()
        
        # Initialize strength columns
        for prefix in ["home", "away"]:
            df[f"{prefix}_win_rate"] = np.nan
            df[f"{prefix}_avg_goals"] = np.nan
            df[f"{prefix}_avg_conceded"] = np.nan
        
        for idx in range(len(df)):
            match = df.iloc[idx]
            home_team = match["home_team"]
            away_team = match["away_team"]
            match_date = match["date"]
            
            # Get historical data before this match
            historical = df[df["date"] < match_date]
            
            # Home team strength (all matches, not just home)
            home_matches = historical[
                (historical["home_team"] == home_team) | 
                (historical["away_team"] == home_team)
            ]
            
            if len(home_matches) >= 10:  # Need at least 10 matches
                home_stats = self._calculate_team_strength(home_matches, home_team)
                df.at[idx, "home_win_rate"] = home_stats["win_rate"]
                df.at[idx, "home_avg_goals"] = home_stats["avg_goals"]
                df.at[idx, "home_avg_conceded"] = home_stats["avg_conceded"]
            
            # Away team strength
            away_matches = historical[
                (historical["home_team"] == away_team) | 
                (historical["away_team"] == away_team)
            ]
            
            if len(away_matches) >= 10:
                away_stats = self._calculate_team_strength(away_matches, away_team)
                df.at[idx, "away_win_rate"] = away_stats["win_rate"]
                df.at[idx, "away_avg_goals"] = away_stats["avg_goals"]
                df.at[idx, "away_avg_conceded"] = away_stats["avg_conceded"]
        
        return df
    
    def _calculate_team_strength(self, matches, team):
        """Calculate overall team strength metrics"""
        wins = 0
        total_goals = 0
        total_conceded = 0
        
        for _, match in matches.iterrows():
            is_home = match["home_team"] == team
            
            if is_home:
                total_goals += match["home_score"]
                total_conceded += match["away_score"]
                if match["result"] == "H":
                    wins += 1
            else:
                total_goals += match["away_score"]
                total_conceded += match["home_score"]
                if match["result"] == "A":
                    wins += 1
        
        num_matches = len(matches)
        return {
            "win_rate": wins / num_matches if num_matches > 0 else 0,
            "avg_goals": total_goals / num_matches if num_matches > 0 else 0,
            "avg_conceded": total_conceded / num_matches if num_matches > 0 else 0
        }
    
    def _create_context_features(self, df):
        """Create contextual features (home advantage, etc.)"""
        df = df.copy()
        
        # Days since last match for each team
        df["home_days_rest"] = np.nan
        df["away_days_rest"] = np.nan
        
        for idx in range(len(df)):
            match = df.iloc[idx]
            home_team = match["home_team"]
            away_team = match["away_team"]
            match_date = match["date"]
            
            # Find last match for each team
            previous_matches = df[df["date"] < match_date]
            
            home_last = previous_matches[
                (previous_matches["home_team"] == home_team) | 
                (previous_matches["away_team"] == home_team)
            ]
            if len(home_last) > 0:
                days_rest = (match_date - home_last.iloc[-1]["date"]).days
                df.at[idx, "home_days_rest"] = days_rest
            
            away_last = previous_matches[
                (previous_matches["home_team"] == away_team) | 
                (previous_matches["away_team"] == away_team)
            ]
            if len(away_last) > 0:
                days_rest = (match_date - away_last.iloc[-1]["date"]).days
                df.at[idx, "away_days_rest"] = days_rest
        
        # Derived features
        df["form_diff"] = df["home_form_points"] - df["away_form_points"]
        df["strength_diff"] = df["home_win_rate"] - df["away_win_rate"]
        df["goals_diff"] = df["home_avg_goals"] - df["away_avg_goals"]
        
        return df
    
    def get_feature_columns(self):
        """Return list of feature column names"""
        return [
            # Home form features
            "home_form_points", "home_form_goals_scored", "home_form_goals_conceded",
            "home_form_wins", "home_form_draws", "home_form_losses",
            # Away form features
            "away_form_points", "away_form_goals_scored", "away_form_goals_conceded",
            "away_form_wins", "away_form_draws", "away_form_losses",
            # H2H features
            "h2h_home_wins", "h2h_draws", "h2h_away_wins",
            "h2h_home_goals_avg", "h2h_away_goals_avg",
            # Strength features
            "home_win_rate", "home_avg_goals", "home_avg_conceded",
            "away_win_rate", "away_avg_goals", "away_avg_conceded",
            # Context features
            "home_days_rest", "away_days_rest",
            # Derived features
            "form_diff", "strength_diff", "goals_diff"
        ]


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append("../..")
    from src.data.collector import FootballDataCollector
    from src.data.preprocessor import DataPreprocessor
    
    collector = FootballDataCollector()
    df = collector.load_data()
    
    if not df.empty:
        preprocessor = DataPreprocessor()
        df_clean = preprocessor.clean_data(df)
        df_encoded = preprocessor.encode_results(df_clean)
        
        engineer = FeatureEngineer()
        df_features = engineer.create_all_features(df_encoded)
        
        print(f"\nFinal dataset: {len(df_features)} matches")
        print(f"Features: {len(engineer.get_feature_columns())}")
        print("\nFeature columns:")
        for col in engineer.get_feature_columns():
            print(f"  - {col}")
        
        print("\nSample features:")
        print(df_features[engineer.get_feature_columns()].head())
