"""
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
import subprocess
import shutil
from config.llm import config as llm_config

# --- Paths ---
OUTPUT_DIR = bootstrap.OUTPUTS_DIR / "reporting"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
OUTPUT_FILE = OUTPUT_DIR / "attribution_analysis_report.md"

MODEL_FILE = bootstrap.OUTPUTS_DIR / "conversion_model.pkl"
PREDICTIONS_FILE = bootstrap.OUTPUTS_DIR / "conversion_predictions.csv"
SHAPLEY_FILE = bootstrap.OUTPUTS_DIR / "shapley_channel_attribution.csv"
SENTIMENT_FILE = bootstrap.PROCESSED_DIR / "llm_sentiment_features.csv"
INTENT_FILE = bootstrap.PROCESSED_DIR / "llm_intent_features.csv"

# --- Load data ---
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


def generate_recommendations(model_insights, attribution_insights, behavior_insights, visuals=None, model_override=None):
    """Generate concise recommendations using the configured LLM (falls back to template).

    Returns a markdown-formatted string containing recommendation sections.
    """
    # Prepare compact context
    acc = model_insights.get('accuracy')
    roc = model_insights.get('roc_auc')
    preds = model_insights.get('predictions')

    top_channels = attribution_insights.get('top_channels')
    try:
        top_channels_text = top_channels[['channel', 'attribution_percentage', 'total_touchpoints']].to_string(index=False)
    except Exception:
        top_channels_text = str(top_channels)

    sentiment = behavior_insights.get('sentiment', {})
    top_intents = behavior_insights.get('top_intents')
    try:
        top_intents_text = top_intents.to_string()
    except Exception:
        top_intents_text = str(top_intents)

    visuals_text = ", ".join(visuals) if visuals else "none"

    # Pre-format numeric context to avoid f-string complexity
    acc_text = f"{acc:.2%}" if acc is not None else "N/A"
    roc_text = f"{roc:.2f}" if roc is not None else "N/A"
    preds_text = str(preds)
    sent_mean = sentiment.get('mean')
    sent_mean_text = f"{sent_mean:.2f}" if sent_mean is not None else "N/A"
    sent_pos = sentiment.get('positive')
    sent_pos_text = f"{sent_pos:.2%}" if sent_pos is not None else "N/A"
    sent_neg = sentiment.get('negative')
    sent_neg_text = f"{sent_neg:.2%}" if sent_neg is not None else "N/A"

    prompt = f"""
You are an expert marketing analyst. Given the analytics context below, produce concise, actionable recommendations in markdown under these headings: Channel Optimization, Content Strategy, Model Improvements, and Quick Experiments (1-2 items).

Context:
- Model accuracy: {acc_text}
- ROC AUC: {roc_text}
- Total predictions: {preds_text}

Top channels (channel / attribution % / touchpoints):
{top_channels_text}

Sentiment summary: mean={sent_mean_text} positive={sent_pos_text} negative={sent_neg_text}

Top intents:
{top_intents_text}

Visualizations available: {visuals_text}

Keep recommendations short (2-4 bullets per section), practical, and prioritized. Respond with markdown only.
"""

    # Use configured model for report recommendations (allow override)
    model_name = model_override or llm_config.get_model("report_recommendation")

    # Fallback generator
    def fallback():
        lines = []
        lines.append("1. Channel Optimization:")
        if isinstance(top_channels, pd.DataFrame) and not top_channels.empty:
            top = list(top_channels['channel'].head(3).astype(str))
            lines.append(f"   - Prioritize channels: {', '.join(top)}; monitor lower-attribution high-touchpoint channels.")
        else:
            lines.append("   - Focus on channels with highest attribution and investigate low-attribution high-touchpoint channels.")

        lines.append("")
        lines.append("2. Content Strategy:")
        if top_intents_text:
            lines.append(f"   - Create content targeting top intents: {top_intents_text.splitlines()[0]}")
        else:
            lines.append("   - Align messaging to common user goals identified in intent analysis.")

        lines.append("")
        lines.append("3. Model Improvements:")
        lines.append("   - Monitor model drift and collect additional features for underperforming segments.")
        lines.append("")
        lines.append("4. Quick Experiments:")
        lines.append("   - Run A/B test on creative for top channel vs top intent audience.")
        return "\n".join(lines)

    # Try calling Ollama via configured CLI; fall back on template
    try:
        if not shutil.which("ollama"):
            if llm_config.fail_fast():
                raise RuntimeError("Ollama CLI not found and fail_fast=True")
            logger.info("Ollama CLI not found — using fallback recommendations")
            return fallback()

        cmd = llm_config.build_cli_command(model_name, prompt)
        res = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30, encoding="utf-8", errors="replace")
        out = res.stdout.strip()
        if not out:
            return fallback()
        return out
    except Exception as e:
        logger.warning(f"LLM recommend generation failed: {e}")
        return fallback()

# Load and analyze all data
model, df_preds, df_shapley, df_sentiment, df_intent = load_data()
model_insights = analyze_model_performance(model, df_preds)
attribution_insights = analyze_attribution(df_shapley)
behavior_insights = analyze_user_behavior(df_sentiment, df_intent)

def generate_markdown_report(model_insights, attribution_insights, behavior_insights):
    """Generate a comprehensive markdown report"""
    # Generate dynamic recommendations via LLM (if available)
    try:
        recommendations_text = generate_recommendations(model_insights, attribution_insights, behavior_insights)
    except Exception as e:
        logger.warning(f"LLM recommendations generation failed: {e}")
        # Fallback recommendations if LLM generation fails
        recommendations_text = """
1. Channel Optimization:
    - Focus on top performing channels while maintaining presence in others

2. Content Strategy:
    - Address common user intents in marketing materials

3. Model Improvements:
    - Continue monitoring model performance
"""

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

{recommendations_text}

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
