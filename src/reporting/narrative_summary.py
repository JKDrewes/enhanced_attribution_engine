"""
narrative_summary.py
Generates a comprehensive narrative summary of model results including:
- Logistic Regression model performance and assumptions
- Shapley attribution analysis
- Channel performance metrics
- User behavior insights from sentiment and intent
"""

# Ensure bootstrap is importable when running directly from this folder
import sys
from pathlib import Path
try:
    import bootstrap
except Exception:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    import bootstrap

import numpy as np
import pandas as pd
from src.utils.logger import logger
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score

# --- Paths ---
OUTPUT_DIR = bootstrap.OUTPUTS_DIR / "reporting"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
OUTPUT_FILE = OUTPUT_DIR / "attribution_analysis_report.md"

MODEL_FILE = bootstrap.OUTPUTS_DIR / "conversion_model.pkl"
PREDICTIONS_FILE = bootstrap.OUTPUTS_DIR / "conversion_predictions.csv"
SHAPLEY_FILE = bootstrap.OUTPUTS_DIR / "shapley_channel_attribution.csv"
SENTIMENT_FILE = bootstrap.PROCESSED_DIR / "llm_sentiment_features.csv"
INTENT_FILE = bootstrap.PROCESSED_DIR / "llm_intent_features.csv"

# --- Load and analyze data ---
def load_data():
    """Load all required data files"""
    logger.info("Loading model and predictions...")
    model = joblib.load(MODEL_FILE)
    df_preds = pd.read_csv(PREDICTIONS_FILE)
    
    logger.info("Loading attribution and features...")
    df_shapley = pd.read_csv(SHAPLEY_FILE)
    df_sentiment = pd.read_csv(SENTIMENT_FILE)
    df_intent = pd.read_csv(INTENT_FILE)
    
    return model, df_preds, df_shapley, df_sentiment, df_intent

def analyze_model_performance(model, df_preds):
    """Analyze logistic regression model performance"""
    y_true = df_preds["y_true"]
    y_prob = df_preds["y_pred"]
    y_pred = (y_prob >= 0.5).astype(int)
    
    accuracy = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    feature_names = model.named_steps['preprocessor'].get_feature_names_out()
    coef = model.named_steps['classifier'].coef_[0]
    top_features = pd.DataFrame({
        'feature': feature_names,
        'importance': np.abs(coef)
    }).nlargest(5, 'importance')
    
    return {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'top_features': top_features,
        'predictions': len(y_true),
        'conversions': sum(y_true)
    }

def analyze_attribution(df_shapley):
    """Analyze Shapley attribution results"""
    top_channels = df_shapley.nlargest(3, 'attribution_percentage')
    correlations = df_shapley[['attribution_percentage', 'total_touchpoints', 'avg_pred_value']].corr()
    
    return {
        'top_channels': top_channels,
        'correlations': correlations,
        'total_touchpoints': df_shapley['total_touchpoints'].sum()
    }

def analyze_user_behavior(df_sentiment, df_intent):
    """Analyze sentiment and intent patterns"""
    sentiment_stats = {
        'mean': df_sentiment['sentiment_score'].mean(),
        'positive': (df_sentiment['sentiment_score'] > 0).mean(),
        'negative': (df_sentiment['sentiment_score'] < 0).mean()
    }
    
    intent_cols = [col for col in df_intent.columns if col.startswith('intent_')]
    top_intents = df_intent[intent_cols].sum().nlargest(5)
    
    return {
        'sentiment': sentiment_stats,
        'top_intents': top_intents
    }

# Load and analyze all data
model, df_preds, df_shapley, df_sentiment, df_intent = load_data()
model_insights = analyze_model_performance(model, df_preds)
attribution_insights = analyze_attribution(df_shapley)
behavior_insights = analyze_user_behavior(df_sentiment, df_intent)

def generate_markdown_report(model_insights, attribution_insights, behavior_insights):
    """Generate a comprehensive markdown report"""
    
    report = f"""# Marketing Attribution Analysis Report

## Model Performance Summary
- **Accuracy**: {model_insights['accuracy']:.2%}
- **ROC AUC Score**: {model_insights['roc_auc']:.2f}
- **Total Predictions**: {model_insights['predictions']}
- **Total Conversions**: {model_insights['conversions']}

### Key Model Features
The most influential features in predicting conversions are:
{model_insights['top_features'].to_string()}

## Channel Attribution Analysis
Top performing channels by attribution percentage:
{attribution_insights['top_channels'][['channel', 'attribution_percentage', 'total_touchpoints']].to_string()}

### Channel Performance Metrics
- Total touchpoints across all channels: {attribution_insights['total_touchpoints']:,}
- Correlation between attribution and touchpoints: {attribution_insights['correlations'].loc['attribution_percentage', 'total_touchpoints']:.2f}

## User Behavior Insights

### Sentiment Analysis
- Average sentiment score: {behavior_insights['sentiment']['mean']:.2f}
- Positive sentiment ratio: {behavior_insights['sentiment']['positive']:.2%}
- Negative sentiment ratio: {behavior_insights['sentiment']['negative']:.2%}

### Top User Intents
The most common user intents identified:
{behavior_insights['top_intents'].to_string()}

## Recommendations

1. Channel Optimization:
   - Focus on top performing channels while maintaining presence in others
   - Investigate high-touchpoint channels with lower attribution

2. Content Strategy:
   - Address common user intents in marketing materials
   - Leverage positive sentiment patterns in messaging

3. Model Improvements:
   - Continue monitoring model performance
   - Consider collecting additional features for key touchpoints

## Visualizations
Please refer to the accompanying visualization report for detailed graphs and charts.
"""
    
    return report

# Generate and save the report
logger.info("Generating comprehensive markdown report...")
report_text = generate_markdown_report(model_insights, attribution_insights, behavior_insights)

with open(OUTPUT_FILE, "w") as f:
    f.write(report_text)

logger.info(f"Report saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    logger.info("Running narrative summary generation directly")
