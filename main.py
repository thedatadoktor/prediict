"""
Main Execution Script
Orchestrates data collection, model training, prediction, and backtesting
"""
import argparse
import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data.collector import FootballDataCollector
from src.data.preprocessor import DataPreprocessor
from src.features.engineer import FeatureEngineer
from src.models.xgboost_model import XGBoostModel
from src.models.lightgbm_model import LightGBMModel
from src.models.ensemble import EnsembleModel
from src.prediction.predictor import MatchPredictor
from src.prediction.odds_analyzer import OddsAnalyzer
from src.evaluation.backtester import Backtester
import config


def collect_data(args):
    """Collect historical match data"""
    print("\n" + "="*80)
    print("COLLECTING DATA")
    print("="*80 + "\n")
    
    collector = FootballDataCollector()
    
    # Parse leagues and seasons
    leagues = args.leagues.split(",") if args.leagues else ["PL"]
    seasons = [int(s) for s in args.seasons.split(",")] if args.seasons else [2021, 2022, 2023]
    
    print(f"Leagues: {', '.join(leagues)}")
    print(f"Seasons: {', '.join(map(str, seasons))}\n")
    
    # Collect data
    df = collector.collect_multiple_seasons(leagues, seasons)
    
    if df.empty:
        print("No data collected!")
        return
    
    # Save data
    collector.save_data(df)
    
    print(f"\nSuccessfully collected {len(df)} matches!")


def train_models(args):
    """Train prediction models"""
    print("\n" + "="*80)
    print("TRAINING MODELS")
    print("="*80 + "\n")
    
    # Load data
    collector = FootballDataCollector()
    df = collector.load_data()
    
    if df.empty:
        print("No data found! Please run --collect first.")
        return
    
    # Preprocess
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.clean_data(df)
    df_encoded = preprocessor.encode_results(df_clean)
    
    # Create features
    engineer = FeatureEngineer()
    df_features = engineer.create_all_features(df_encoded)
    
    print(f"\nDataset: {len(df_features)} matches with {len(engineer.get_feature_columns())} features")
    
    # Split data
    train_df, test_df = preprocessor.split_train_test(df_features, test_size=0.2, by_date=True)
    
    # Prepare training data
    feature_cols = engineer.get_feature_columns()
    X_train = train_df[feature_cols]
    y_train = train_df["result"].map({"H": 0, "D": 1, "A": 2}).values
    
    X_test = test_df[feature_cols]
    y_test = test_df["result"].map({"H": 0, "D": 1, "A": 2}).values
    
    # Train XGBoost
    print("\n" + "-"*80)
    xgb_model = XGBoostModel()
    xgb_model.train(X_train, y_train)
    xgb_results = xgb_model.evaluate(X_test, y_test)
    print(f"\nXGBoost Test Accuracy: {xgb_results['accuracy']:.2%}")
    print(f"XGBoost Log Loss: {xgb_results['log_loss']:.3f}")
    
    # Train LightGBM
    print("\n" + "-"*80)
    lgb_model = LightGBMModel()
    lgb_model.train(X_train, y_train)
    lgb_results = lgb_model.evaluate(X_test, y_test)
    print(f"\nLightGBM Test Accuracy: {lgb_results['accuracy']:.2%}")
    print(f"LightGBM Log Loss: {lgb_results['log_loss']:.3f}")
    
    # Create ensemble
    print("\n" + "-"*80)
    print("\nCreating ensemble model...")
    ensemble = EnsembleModel(
        models=[xgb_model, lgb_model],
        weights=[0.5, 0.5],
        name="ensemble_model"
    )
    ensemble_results = ensemble.evaluate(X_test, y_test)
    print(f"Ensemble Test Accuracy: {ensemble_results['accuracy']:.2%}")
    print(f"Ensemble Log Loss: {ensemble_results['log_loss']:.3f}")
    
    # Save models
    print("\nSaving models...")
    xgb_model.save_model()
    lgb_model.save_model()
    
    # Save feature engineer settings
    import joblib
    joblib.dump(engineer, config.MODELS_DIR / "feature_engineer.joblib")
    print("Feature engineer saved")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80 + "\n")


def make_predictions(args):
    """Make predictions for upcoming matches"""
    print("\n" + "="*80)
    print("MAKING PREDICTIONS")
    print("="*80 + "\n")
    
    # Load models
    import joblib
    
    try:
        xgb_model = XGBoostModel().load_model()
        lgb_model = LightGBMModel().load_model()
        engineer = joblib.load(config.MODELS_DIR / "feature_engineer.joblib")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run --train first to train the models.")
        return
    
    # Create ensemble
    ensemble = EnsembleModel(
        models=[xgb_model, lgb_model],
        weights=[0.5, 0.5]
    )
    
    # Load historical data
    collector = FootballDataCollector()
    historical_data = collector.load_data()
    
    if historical_data.empty:
        print("No historical data found!")
        return
    
    # Preprocess historical data
    preprocessor = DataPreprocessor()
    historical_data = preprocessor.clean_data(historical_data)
    historical_data = preprocessor.encode_results(historical_data)
    
    # Get upcoming matches
    leagues = args.leagues.split(",") if args.leagues else ["PL"]
    days_ahead = args.days or 7
    
    all_upcoming = []
    for league in leagues:
        upcoming = collector.get_upcoming_matches(league, days_ahead=days_ahead)
        if not upcoming.empty:
            all_upcoming.append(upcoming)
    
    if not all_upcoming:
        print(f"No upcoming matches found in the next {days_ahead} days.")
        return
    
    upcoming_matches = pd.concat(all_upcoming, ignore_index=True)
    print(f"Found {len(upcoming_matches)} upcoming matches\n")
    
    # Make predictions
    predictor = MatchPredictor(ensemble, engineer)
    predictions = predictor.predict_matches(upcoming_matches, historical_data)
    
    # Filter high confidence predictions
    high_conf = predictor.filter_high_confidence(predictions)
    
    # Display predictions
    print("\n" + "="*80)
    print("HIGH CONFIDENCE PREDICTIONS")
    print("="*80 + "\n")
    
    pred_df = predictor.to_dataframe(high_conf)
    if not pred_df.empty:
        print(pred_df.to_string(index=False))
    else:
        print("No high confidence predictions found.")
    
    # Note about odds
    print("\n" + "="*80)
    print("NOTE: To identify value bets, add bookmaker odds data")
    print("See odds_analyzer.py for implementation details")
    print("="*80 + "\n")


def run_backtest(args):
    """Run backtest on historical data"""
    print("\n" + "="*80)
    print("RUNNING BACKTEST")
    print("="*80 + "\n")
    
    # Load models
    import joblib
    
    try:
        xgb_model = XGBoostModel().load_model()
        lgb_model = LightGBMModel().load_model()
        engineer = joblib.load(config.MODELS_DIR / "feature_engineer.joblib")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run --train first to train the models.")
        return
    
    # Create ensemble
    ensemble = EnsembleModel(
        models=[xgb_model, lgb_model],
        weights=[0.5, 0.5]
    )
    
    # Load and prepare data
    collector = FootballDataCollector()
    df = collector.load_data()
    
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.clean_data(df)
    df_encoded = preprocessor.encode_results(df_clean)
    df_features = engineer.create_all_features(df_encoded)
    
    # Run backtest
    backtester = Backtester(ensemble, engineer)
    
    start_date = args.start_date
    end_date = args.end_date
    
    backtest_results = backtester.backtest(
        df_features,
        start_date=start_date,
        end_date=end_date
    )
    
    # Simulate betting
    betting_results = backtester.simulate_betting(
        backtest_results,
        min_confidence=0.55,
        bankroll=1000,
        stake_per_bet=10
    )
    
    # Print summary
    backtester.print_summary(backtest_results, betting_results)


def main():
    parser = argparse.ArgumentParser(
        description="Football Match Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect data for Premier League
  python main.py --collect --leagues PL --seasons 2021,2022,2023
  
  # Train models
  python main.py --train
  
  # Make predictions for upcoming matches
  python main.py --predict --leagues PL --days 7
  
  # Run backtest
  python main.py --backtest --start-date 2023-01-01 --end-date 2024-01-01
        """
    )
    
    # Main actions
    parser.add_argument("--collect", action="store_true", help="Collect historical data")
    parser.add_argument("--train", action="store_true", help="Train prediction models")
    parser.add_argument("--predict", action="store_true", help="Predict upcoming matches")
    parser.add_argument("--backtest", action="store_true", help="Run backtest on historical data")
    
    # Options
    parser.add_argument("--leagues", type=str, help="Comma-separated league codes (e.g., PL,PD,SA)")
    parser.add_argument("--seasons", type=str, help="Comma-separated season years (e.g., 2021,2022,2023)")
    parser.add_argument("--days", type=int, help="Days ahead for upcoming matches (default: 7)")
    parser.add_argument("--start-date", type=str, help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="Backtest end date (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    # Execute actions
    if args.collect:
        collect_data(args)
    
    if args.train:
        train_models(args)
    
    if args.predict:
        make_predictions(args)
    
    if args.backtest:
        run_backtest(args)
    
    if not any([args.collect, args.train, args.predict, args.backtest]):
        parser.print_help()


if __name__ == "__main__":
    main()
