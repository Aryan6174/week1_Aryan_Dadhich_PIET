"""
Visualization utilities
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def plot_correlation_heatmap(df):
    """Create correlation heatmap using plotly"""
    corr = df.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title='Correlation Heatmap',
        xaxis_title='Features',
        yaxis_title='Features',
        height=600,
        width=800
    )
    
    return fig

def plot_distributions(df, columns):
    """Create distribution plots"""
    n_cols = 3
    n_rows = (len(columns) + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=columns
    )
    
    for idx, col in enumerate(columns):
        row = idx // n_cols + 1
        col_pos = idx % n_cols + 1
        
        fig.add_trace(
            go.Histogram(x=df[col], name=col, showlegend=False),
            row=row, col=col_pos
        )
    
    fig.update_layout(height=300*n_rows, showlegend=False, title_text="Feature Distributions")
    return fig

def plot_pca_clusters(pca_df, labels, title='PCA Cluster Visualization'):
    """Create PCA cluster visualization"""
    fig = px.scatter(
        pca_df, x='PC1', y='PC2',
        color=labels.astype(str),
        title=title,
        labels={'color': 'Cluster'},
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))
    fig.update_layout(height=600, width=800)
    
    return fig

def plot_elbow_silhouette(k_values, inertias, silhouette_scores):
    """Create elbow and silhouette plots"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Elbow Method', 'Silhouette Score')
    )
    
    # Elbow plot
    fig.add_trace(
        go.Scatter(x=k_values, y=inertias, mode='lines+markers', name='Inertia'),
        row=1, col=1
    )
    
    # Silhouette plot
    fig.add_trace(
        go.Scatter(x=k_values, y=silhouette_scores, mode='lines+markers', 
                  name='Silhouette Score', line=dict(color='red')),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Number of Clusters (K)", row=1, col=1)
    fig.update_xaxes(title_text="Number of Clusters (K)", row=1, col=2)
    fig.update_yaxes(title_text="Inertia", row=1, col=1)
    fig.update_yaxes(title_text="Silhouette Score", row=1, col=2)
    
    fig.update_layout(height=400, showlegend=False)
    
    return fig

def plot_model_comparison(results_df):
    """Create model comparison plot"""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    fig = go.Figure()
    
    for metric in metrics:
        fig.add_trace(go.Bar(
            name=metric,
            x=results_df['Model'],
            y=results_df[metric],
            text=results_df[metric].round(4),
            textposition='auto',
        ))
    
    fig.update_layout(
        title='Model Performance Comparison',
        xaxis_title='Models',
        yaxis_title='Score',
        barmode='group',
        height=500,
        yaxis=dict(range=[0, 1.1])
    )
    
    return fig

def plot_feature_importance(importance_df, title='Feature Importance'):
    """Create feature importance plot"""
    fig = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title=title,
        color='Importance',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(height=500, yaxis={'categoryorder':'total ascending'})
    
    return fig

def plot_confusion_matrix(cm, model_name):
    """Create confusion matrix heatmap"""
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=[f'Predicted {i}' for i in range(len(cm))],
        y=[f'Actual {i}' for i in range(len(cm))],
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 16},
    ))
    
    fig.update_layout(
        title=f'Confusion Matrix - {model_name}',
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        height=400,
        width=400
    )
    
    return fig

def plot_priority_countries(countries_df, top_n=10):
    """Create priority countries visualization"""
    top_countries = countries_df.head(top_n)
    
    fig = go.Figure(go.Bar(
        x=top_countries['Priority_Score'],
        y=top_countries['country'],
        orientation='h',
        marker=dict(
            color=top_countries['Priority_Score'],
            colorscale='Reds',
            showscale=True
        ),
        text=top_countries['Priority_Score'].round(4),
        textposition='auto',
    ))
    
    fig.update_layout(
        title=f'Top {top_n} Priority Countries for Aid',
        xaxis_title='Priority Score',
        yaxis_title='Country',
        height=500,
        yaxis={'categoryorder':'total ascending'}
    )
    
    return fig