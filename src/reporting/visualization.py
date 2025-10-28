"""
visualization.py
Generates a comprehensive set of visualizations for the attribution analysis:
1. Model Performance:
   - ROC and Precision-Recall curves
   - Feature importance plot
   - Prediction distribution
2. Channel Attribution:
   - Attribution percentages
   - Touchpoint distribution
   - Attribution vs touchpoints scatter
3. User Behavior:
   - Sentiment distribution
   - Intent frequency
   - Channel-sentiment relationship
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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)
from utils.logger import logger
import joblib

# Set style for all plots
sns.set_style("whitegrid")
sns.set_palette("husl")

# --- Paths ---
OUTPUT_DIR = bootstrap.OUTPUTS_DIR / "reporting" / "figures"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

MODEL_FILE = bootstrap.OUTPUTS_DIR / "conversion_model.pkl"
PREDICTIONS_FILE = bootstrap.OUTPUTS_DIR / "conversion_predictions.csv"
SHAPLEY_FILE = bootstrap.OUTPUTS_DIR / "shapley_channel_attribution.csv"
SENTIMENT_FILE = bootstrap.PROCESSED_DIR / "llm_sentiment_features.csv"
INTENT_FILE = bootstrap.PROCESSED_DIR / "llm_intent_features.csv"

def plot_model_performance(model, df_preds, output_dir):
    """Create model performance visualizations"""
    y_true = df_preds["y_true"]
    y_prob = df_preds["y_pred"]
    y_pred = (y_prob >= 0.5).astype(int)
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    axes[0,0].plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.2f})')
    axes[0,0].plot([0, 1], [0, 1], 'k--')
    axes[0,0].set_xlabel('False Positive Rate')
    axes[0,0].set_ylabel('True Positive Rate')
    axes[0,0].set_title('ROC Curve')
    axes[0,0].legend()
    
    # 2. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    avg_prec = average_precision_score(y_true, y_prob)
    axes[0,1].plot(recall, precision, label=f'AP = {avg_prec:.2f}')
    axes[0,1].set_xlabel('Recall')
    axes[0,1].set_ylabel('Precision')
    axes[0,1].set_title('Precision-Recall Curve')
    axes[0,1].legend()
    
    # 3. Feature Importance
    feature_names = model.named_steps['preprocessor'].get_feature_names_out()
    coef = model.named_steps['classifier'].coef_[0]
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': np.abs(coef)
    }).nlargest(10, 'importance')
    
    sns.barplot(data=importance_df, x='importance', y='feature', ax=axes[1,0])
    axes[1,0].set_title('Top 10 Feature Importance')
    
    # 4. Prediction Distribution
    sns.histplot(data=y_prob, bins=30, ax=axes[1,1])
    axes[1,1].axvline(0.5, color='r', linestyle='--', alpha=0.5)
    axes[1,1].set_title('Prediction Distribution')
    axes[1,1].set_xlabel('Predicted Probability')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

# Load all data
logger.info("Loading model and predictions...")
model = joblib.load(MODEL_FILE)
df_preds = pd.read_csv(PREDICTIONS_FILE)

def plot_attribution_analysis(df_shapley, output_dir):
    """Create attribution analysis visualizations"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Attribution and touchpoints combined view
    df_plot = df_shapley.sort_values('attribution_percentage', ascending=True)
    x = np.arange(len(df_plot))
    width = 0.35
    
    ax1.bar(x - width/2, df_plot['attribution_percentage'], width, label='Attribution %')
    ax1.bar(x + width/2, df_plot['total_touchpoints']/df_plot['total_touchpoints'].max()*20, 
            width, label='Relative Touchpoints')
    ax1.set_ylabel('Percentage / Relative Scale')
    ax1.set_title('Channel Attribution and Touchpoint Distribution')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_plot['channel'], rotation=45, ha='right')
    ax1.legend()
    
    # Correlation scatterplot
    sns.scatterplot(
        data=df_shapley,
        x='total_touchpoints',
        y='attribution_percentage',
        size='avg_pred_value',
        sizes=(100, 400),
        alpha=0.6,
        ax=ax2
    )
    
    for _, row in df_shapley.iterrows():
        ax2.annotate(
            row['channel'],
            (row['total_touchpoints'], row['attribution_percentage']),
            xytext=(5, 5), textcoords='offset points'
        )
    
    ax2.set_title('Attribution vs Touchpoints')
    ax2.set_xlabel('Total Touchpoints')
    ax2.set_ylabel('Attribution Percentage')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'attribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_user_behavior(df_sentiment, df_intent, df_shapley, output_dir):
    """Create user behavior visualizations"""
    fig = plt.figure(figsize=(15, 10))
    gs = plt.GridSpec(2, 2, figure=fig)
    
    # 1. Sentiment Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    sns.histplot(data=df_sentiment, x='sentiment_score', bins=30, kde=True, ax=ax1)
    ax1.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    ax1.set_title('Sentiment Distribution')
    
    # 2. Top Intents
    ax2 = fig.add_subplot(gs[0, 1])
    intent_cols = [col for col in df_intent.columns if col.startswith('intent_')]
    top_intents = df_intent[intent_cols].sum().nlargest(5)
    sns.barplot(x=top_intents.values, y=[i.replace('intent_', '').replace('_', ' ') 
                                        for i in top_intents.index], ax=ax2)
    ax2.set_title('Top 5 User Intents')
    
    # 3. Sentiment Distribution by Value
    ax3 = fig.add_subplot(gs[1, :])
    sentiment_bins = pd.cut(df_sentiment['sentiment_score'], bins=10)
    sentiment_dist = pd.DataFrame({
        'count': sentiment_bins.value_counts(),
        'sentiment_range': sentiment_bins.value_counts().index
    }).sort_values('sentiment_range')
    
    sns.barplot(data=sentiment_dist, x='sentiment_range', y='count',
                ax=ax3, color='skyblue', alpha=0.6)
    ax3.set_title('Detailed Sentiment Distribution')
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'user_behavior.png', dpi=300, bbox_inches='tight')
    plt.close()

# Load all data and generate visualizations
logger.info("Loading data files...")
model = joblib.load(MODEL_FILE)
df_preds = pd.read_csv(PREDICTIONS_FILE)
df_shapley = pd.read_csv(SHAPLEY_FILE)
df_sentiment = pd.read_csv(SENTIMENT_FILE)
df_intent = pd.read_csv(INTENT_FILE)

logger.info("Generating visualizations...")
plot_model_performance(model, df_preds, OUTPUT_DIR)
plot_attribution_analysis(df_shapley, OUTPUT_DIR)
plot_user_behavior(df_sentiment, df_intent, df_shapley, OUTPUT_DIR)

logger.info(f"All visualizations saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    logger.info("Running visualization generation directly")
