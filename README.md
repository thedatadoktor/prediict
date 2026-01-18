# Football Match Prediction System

A comprehensive machine learning system for predicting football match outcomes and identifying value bets.

## Features

- **Data Collection**: Automated data fetching from football APIs
- **Feature Engineering**: Advanced statistical features (form, head-to-head, home/away)
- **ML Models**: Multiple models (XGBoost, LightGBM, Neural Networks)
- **Value Betting**: Compare predictions against bookmaker odds
- **Backtesting**: Historical performance evaluation

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Get a free API key from [Football-Data.org](https://www.football-data.org/client/register)

4. Create `.env` file:
   ```bash
   cp .env.example .env
   ```
   Add your API key to `.env`

## Usage

### Collect Historical Data
```bash
python main.py --collect --leagues PL,PD,SA --seasons 2020,2021,2022,2023
```

### Train Models
```bash
python main.py --train
```

### Make Predictions
```bash
python main.py --predict
```

### Backtest Performance
```bash
python main.py --backtest --start-date 2023-01-01 --end-date 2024-01-01
```

## Project Structure

```
prediict/
├── src/
│   ├── data/
│   │   ├── collector.py       # Data collection from APIs
│   │   └── preprocessor.py    # Data cleaning and preparation
│   ├── features/
│   │   └── engineer.py        # Feature engineering
│   ├── models/
│   │   ├── base_model.py      # Base model class
│   │   ├── xgboost_model.py   # XGBoost implementation
│   │   ├── lightgbm_model.py  # LightGBM implementation
│   │   └── ensemble.py        # Model ensemble
│   ├── prediction/
│   │   ├── predictor.py       # Prediction system
│   │   └── odds_analyzer.py   # Value bet identification
│   └── evaluation/
│       └── backtester.py      # Backtesting framework
├── config.py                   # Configuration settings
├── main.py                     # Main execution script
└── requirements.txt
```

## Disclaimer

⚠️ **Important**: This system is for educational and research purposes. Betting involves risk and there are no guarantees of profit. Always gamble responsibly and never bet more than you can afford to lose.

## License

MIT License
