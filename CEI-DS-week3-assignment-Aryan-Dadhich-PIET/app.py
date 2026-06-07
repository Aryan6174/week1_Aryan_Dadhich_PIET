"""
Country Intelligence System - Streamlit Web Application
For HELP International NGO
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import plotly.express as px
import plotly.graph_objects as go

# Import utility functions
from utils.data_loader import load_data, get_data_info, get_summary_stats, check_data_quality
from utils.preprocessing import preprocess_data, scale_features, apply_pca
from utils.clustering import (
    perform_kmeans, perform_dbscan, find_optimal_k, 
    calculate_silhouette, calculate_clustering_metrics
)
from utils.classification import train_all_models, evaluate_models, get_feature_importance, split_data
from utils.visualizations import (
    plot_correlation_heatmap, plot_distributions, plot_pca_clusters,
    plot_elbow_silhouette, plot_model_comparison, plot_feature_importance,
    plot_confusion_matrix, plot_priority_countries
)
import config

# Page configuration
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stAlert {
        background-color: #e3f2fd;
    }
    h1 {
        color: #1976d2;
    }
    h2 {
        color: #424242;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'clustering_done' not in st.session_state:
    st.session_state.clustering_done = False

# Sidebar
# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/globe-earth.png", width=80)
    st.title("🌍 Navigation")
    
    # Developer credit in header
    st.markdown("""
    <div style='text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 10px; border-radius: 10px; margin-bottom: 15px;'>
        <p style='color: white; font-size: 12px; margin: 0; font-weight: bold;'>
            💻 Made by<br>Aryan Dadhich<br>PIET
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.radio(
        "Go to",
        ["🏠 Home", "📊 Data Overview", "🔍 EDA", "🎯 Clustering", 
         "🤖 Classification", "🎯 Recommendations", "📥 Download Results"]
    )
    
    st.markdown("---")
    st.markdown("### About")
    st.info("""
    **HELP International**
    
    Humanitarian NGO committed to fighting poverty and providing aid to underdeveloped countries.
    
    **Budget:** $10 Million
    """)
    
    st.markdown("---")
    st.markdown("### Settings")
    optimal_k = st.slider("Number of Clusters (K)", 2, 10, config.OPTIMAL_CLUSTERS)
    test_size = st.slider("Test Size", 0.1, 0.4, config.TEST_SIZE, 0.05)
# Main content
if page == "🏠 Home":
    st.title(config.APP_TITLE)
    st.markdown(f"## {config.APP_SUBTITLE}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
        <h3>🎯 Objective</h3>
        <p>Identify countries in direst need of aid using AI-powered analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
        <h3>💰 Budget</h3>
        <p>$10 Million available for strategic allocation</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
        <h3>📈 Approach</h3>
        <p>Clustering + Classification + AI Predictions</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("## 📂 Upload Dataset")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload Country-data.csv file"
        )
        
        use_sample = st.checkbox("Or use default dataset (data/Country-data.csv)")
    
    if uploaded_file is not None or use_sample:
        try:
            if uploaded_file is not None:
                df = load_data(uploaded_file=uploaded_file)
            else:
                df = load_data(file_path='data/Country-data.csv')
            
            if df is not None:
                st.session_state.df = df
                st.session_state.data_loaded = True
                
                st.success(f"✅ Data loaded successfully! {df.shape[0]} countries, {df.shape[1]} features")
                
                with col2:
                    st.metric("Countries", df.shape[0])
                    st.metric("Features", df.shape[1])
                
                st.markdown("### Preview")
                st.dataframe(df.head(10), use_container_width=True)
                
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
    
    st.markdown("---")
    
    st.markdown("## 🚀 Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📊 Analysis Capabilities
        - **Exploratory Data Analysis**: Comprehensive statistical analysis
        - **Clustering**: K-Means & DBSCAN algorithms
        - **Classification**: 4 ML models (LR, DT, RF, XGBoost)
        - **Feature Importance**: Identify key development indicators
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 Deliverables
        - **Country Segmentation**: Group countries by development level
        - **Priority Ranking**: Identify countries needing urgent aid
        - **Predictive Models**: Classify new countries automatically
        - **Downloadable Reports**: Export results in CSV/Excel
        """)

elif page == "📊 Data Overview":
    st.title("📊 Data Overview & Quality Check")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload data from the Home page first!")
    else:
        df = st.session_state.df
        
        # Data Info
        st.markdown("## 📋 Dataset Information")
        
        col1, col2, col3, col4 = st.columns(4)
        
        info = get_data_info(df)
        
        with col1:
            st.metric("Total Countries", info['shape'][0])
        with col2:
            st.metric("Total Features", info['shape'][1])
        with col3:
            st.metric("Missing Values", sum(info['missing'].values()))
        with col4:
            st.metric("Duplicates", info['duplicates'])
        
        st.markdown("---")
        
        # Column Information
        st.markdown("## 📑 Feature Descriptions")
        
        feature_info = []
        for col in df.columns:
            if col in config.FEATURE_DESCRIPTIONS:
                feature_info.append({
                    'Feature': col,
                    'Description': config.FEATURE_DESCRIPTIONS[col],
                    'Type': str(df[col].dtype),
                    'Missing': df[col].isnull().sum(),
                    'Unique': df[col].nunique()
                })
        
        st.dataframe(pd.DataFrame(feature_info), use_container_width=True)
        
        st.markdown("---")
        
        # Summary Statistics
        st.markdown("## 📈 Summary Statistics")
        
        numeric_df = df.select_dtypes(include=[np.number])
        st.dataframe(numeric_df.describe().T, use_container_width=True)
        
        st.markdown("---")
        
        # Data Quality
        st.markdown("## ✅ Data Quality Check")
        
        quality_issues = check_data_quality(df)
        
        for issue in quality_issues:
            if "✓" in issue:
                st.success(issue)
            else:
                st.warning(issue)

elif page == "🔍 EDA":
    st.title("🔍 Exploratory Data Analysis")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload data from the Home page first!")
    else:
        df = st.session_state.df
        numeric_df = df.select_dtypes(include=[np.number])
        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Distributions", "🔥 Correlations", "📦 Outliers", "🔑 Key Insights"])
        
        with tab1:
            st.markdown("### Feature Distributions")
            
            selected_features = st.multiselect(
                "Select features to visualize",
                numeric_df.columns.tolist(),
                default=numeric_df.columns.tolist()[:3]
            )
            
            if selected_features:
                fig = plot_distributions(numeric_df, selected_features)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.markdown("### Correlation Analysis")
            
            fig = plot_correlation_heatmap(numeric_df)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("#### Key Correlations")
            corr_matrix = numeric_df.corr()
            
            # Find top correlations
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_pairs.append({
                        'Feature 1': corr_matrix.columns[i],
                        'Feature 2': corr_matrix.columns[j],
                        'Correlation': corr_matrix.iloc[i, j]
                    })
            
            corr_df = pd.DataFrame(corr_pairs).sort_values('Correlation', key=abs, ascending=False)
            st.dataframe(corr_df.head(10), use_container_width=True)
        
        with tab3:
            st.markdown("### Outlier Detection")
            
            selected_feature = st.selectbox("Select feature for boxplot", numeric_df.columns)
            
            fig = go.Figure()
            fig.add_trace(go.Box(y=numeric_df[selected_feature], name=selected_feature))
            fig.update_layout(title=f'Boxplot - {selected_feature}', height=400)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Outlier statistics
            Q1 = numeric_df[selected_feature].quantile(0.25)
            Q3 = numeric_df[selected_feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = numeric_df[(numeric_df[selected_feature] < lower_bound) | 
                                 (numeric_df[selected_feature] > upper_bound)]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Lower Bound", f"{lower_bound:.2f}")
            with col2:
                st.metric("Upper Bound", f"{upper_bound:.2f}")
            with col3:
                st.metric("Outliers", len(outliers))
        
        with tab4:
            st.markdown("### 🔑 Key Insights")
            
            st.markdown("""
            #### Development Indicators Analysis
            
            **Strong Negative Correlations:**
            - Child Mortality ↔ Life Expectancy: High child mortality = Lower life expectancy
            - Child Mortality ↔ Income: Poorer countries have higher child deaths
            
            **Strong Positive Correlations:**
            - Income ↔ GDP per Capita: Wealth indicators align
            - Health Spending ↔ Life Expectancy: Better healthcare = Longer lives
            
            **Distribution Insights:**
            - Life expectancy ranges from 32 to 83 years (51-year gap!)
            - Income disparity: $609 to $125,000 per capita
            - Child mortality: 2.6 to 208 per 1000 births
            
            **Outliers:**
            - Several countries have extremely high GDP per capita
            - Inflation varies dramatically (-4.2% to 104%)
            - Export/import ratios show high variability
            """)

elif page == "🎯 Clustering":
    st.title("🎯 Clustering Analysis")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload data from the Home page first!")
    else:
        df = st.session_state.df
        
        # Preprocess data
        df_numeric, countries = preprocess_data(df, target_col='country')
        df_scaled, scaler = scale_features(df_numeric)
        
        tab1, tab2, tab3, tab4 = st.tabs(["🎲 K-Means", "🔬 DBSCAN", "📉 PCA", "⚖️ Comparison"])
        
        with tab1:
            st.markdown("### K-Means Clustering")
            
            # Find optimal K
            with st.spinner("Finding optimal K..."):
                k_values, inertias, silhouette_scores = find_optimal_k(df_scaled, range(2, 11))
            
            # Plot elbow and silhouette
            fig = plot_elbow_silhouette(k_values, inertias, silhouette_scores)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Perform K-Means with selected K
            kmeans_labels, kmeans_model = perform_kmeans(df_scaled, n_clusters=optimal_k)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Selected K", optimal_k)
                silhouette = calculate_silhouette(df_scaled, kmeans_labels)
                st.metric("Silhouette Score", f"{silhouette:.4f}")
            
            with col2:
                # Cluster distribution
                unique, counts = np.unique(kmeans_labels, return_counts=True)
                cluster_dist = pd.DataFrame({
                    'Cluster': unique,
                    'Count': counts,
                    'Percentage': (counts / len(kmeans_labels) * 100).round(2)
                })
                st.dataframe(cluster_dist, use_container_width=True)
            
            # PCA Visualization
            st.markdown("#### Cluster Visualization (PCA)")
            pca_df, explained_var, pca_model = apply_pca(df_scaled, n_components=2)
            
            fig = plot_pca_clusters(pca_df, kmeans_labels, 
                                   title=f'K-Means Clustering (K={optimal_k})')
            st.plotly_chart(fig, use_container_width=True)
            
            st.info(f"📊 Explained Variance: PC1={explained_var[0]:.2%}, PC2={explained_var[1]:.2%}, Total={sum(explained_var):.2%}")
            
            # Save to session state
            st.session_state.kmeans_labels = kmeans_labels
            st.session_state.kmeans_model = kmeans_model
            st.session_state.pca_df = pca_df
            st.session_state.clustering_done = True
        
        with tab2:
            st.markdown("### DBSCAN Clustering")
            
            col1, col2 = st.columns(2)
            
            with col1:
                eps = st.slider("Epsilon (eps)", 0.5, 5.0, 2.5, 0.1)
            with col2:
                min_samples = st.slider("Min Samples", 2, 10, 3)
            
            if st.button("Run DBSCAN"):
                dbscan_labels, n_clusters, n_noise = perform_dbscan(df_scaled, eps, min_samples)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Clusters Found", n_clusters)
                with col2:
                    st.metric("Noise Points", n_noise)
                with col3:
                    if n_clusters > 1:
                        metrics = calculate_clustering_metrics(df_scaled.values, dbscan_labels)
                        st.metric("Silhouette Score", f"{metrics['silhouette']:.4f}")
                
                # Visualization
                pca_df, _, _ = apply_pca(df_scaled, n_components=2)
                fig = plot_pca_clusters(pca_df, dbscan_labels, 
                                       title='DBSCAN Clustering')
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown("### Principal Component Analysis")
            
            n_components = st.slider("Number of Components", 2, 5, 2)
            
            pca_df_full, explained_var_full, _ = apply_pca(df_scaled, n_components=n_components)
            
            # Scree plot
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[f'PC{i+1}' for i in range(n_components)],
                y=explained_var_full,
                text=[f'{var:.2%}' for var in explained_var_full],
                textposition='auto'
            ))
            fig.update_layout(
                title='PCA Explained Variance',
                xaxis_title='Principal Components',
                yaxis_title='Explained Variance Ratio',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("#### Component Loadings")
            
            # Calculate loadings
            pca_full = apply_pca(df_scaled, n_components=n_components)[2]
            loadings = pd.DataFrame(
                pca_full.components_.T,
                columns=[f'PC{i+1}' for i in range(n_components)],
                index=df_numeric.columns
            )
            
            st.dataframe(loadings.style.background_gradient(cmap='RdBu', axis=0), 
                        use_container_width=True)
        
        with tab4:
            st.markdown("### Clustering Comparison")
            
            if st.session_state.clustering_done:
                kmeans_metrics = calculate_clustering_metrics(
                    df_scaled.values, 
                    st.session_state.kmeans_labels
                )
                
                comparison = pd.DataFrame({
                    'Metric': ['Silhouette Score', 'Calinski-Harabasz', 'Davies-Bouldin'],
                    'K-Means': [
                        f"{kmeans_metrics['silhouette']:.4f}",
                        f"{kmeans_metrics['calinski_harabasz']:.2f}",
                        f"{kmeans_metrics['davies_bouldin']:.4f}"
                    ]
                })
                
                st.dataframe(comparison, use_container_width=True)
                
                st.markdown("""
                #### Interpretation
                
                - **Silhouette Score**: Ranges from -1 to 1. Higher is better (>0.5 = good)
                - **Calinski-Harabasz**: Higher values indicate better clustering
                - **Davies-Bouldin**: Lower values indicate better clustering
                
                ✅ **Recommendation**: Use K-Means clusters for classification task
                """)

elif page == "🤖 Classification":
    st.title("🤖 Classification Models")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload data from the Home page first!")
    elif not st.session_state.clustering_done:
        st.warning("⚠️ Please complete clustering analysis first!")
    else:
        df = st.session_state.df
        df_numeric, countries = preprocess_data(df, target_col='country')
        df_scaled, _ = scale_features(df_numeric)
        
        X = df_scaled
        y = st.session_state.kmeans_labels
        
        tab1, tab2, tab3 = st.tabs(["🎯 Train Models", "📊 Evaluation", "🎨 Feature Importance"])
        
        with tab1:
            st.markdown("### Train Classification Models")
            
            st.info("Using K-Means cluster labels as target variable for supervised learning")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Samples", len(X))
                st.metric("Features", X.shape[1])
            
            with col2:
                st.metric("Target Classes", len(np.unique(y)))
                st.metric("Test Size", f"{test_size*100:.0f}%")
            
            if st.button("🚀 Train All Models", type="primary"):
                with st.spinner("Training models..."):
                    # Split data
                    X_train, X_test, y_train, y_test = split_data(X, y, test_size, config.RANDOM_STATE)
                    
                    # Train models
                    models, predictions = train_all_models(X_train, X_test, y_train, y_test)
                    
                    # Save to session state
                    st.session_state.models = models
                    st.session_state.predictions = predictions
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    st.session_state.models_trained = True
                    
                    st.success("✅ All models trained successfully!")
        
        with tab2:
            st.markdown("### Model Evaluation")
            
            if not st.session_state.models_trained:
                st.warning("⚠️ Please train models first!")
            else:
                predictions = st.session_state.predictions
                y_test = st.session_state.y_test
                
                # Evaluate models
                results_df = evaluate_models(y_test, predictions)
                results_df = results_df.sort_values('F1-Score', ascending=False)
                
                st.markdown("#### Performance Metrics")
                
                # Format for display
                display_df = results_df.copy()
                for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
                    display_df[col] = display_df[col].apply(lambda x: f'{x*100:.2f}%')
                
                st.dataframe(display_df, use_container_width=True)
                
                # Visual comparison
                fig = plot_model_comparison(results_df)
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                
                # Confusion Matrices
                st.markdown("#### Confusion Matrices")
                
                selected_model = st.selectbox("Select Model", list(predictions.keys()))
                
                cm = confusion_matrix(y_test, predictions[selected_model])
                fig = plot_confusion_matrix(cm, selected_model)
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("##### Classification Report")
                    from sklearn.metrics import classification_report
                    report = classification_report(y_test, predictions[selected_model], 
                                                  output_dict=True)
                    st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)
                
                # Best model
                best_model = results_df.iloc[0]['Model']
                best_score = results_df.iloc[0]['F1-Score']
                
                st.success(f"🏆 Best Model: **{best_model}** with F1-Score: **{best_score:.4f}**")
        
        with tab3:
            st.markdown("### Feature Importance Analysis")
            
            if not st.session_state.models_trained:
                st.warning("⚠️ Please train models first!")
            else:
                models = st.session_state.models
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Random Forest")
                    rf_importance = get_feature_importance(
                        models['Random Forest'], 
                        df_numeric.columns
                    )
                    if rf_importance is not None:
                        fig = plot_feature_importance(rf_importance, 'Random Forest Feature Importance')
                        st.plotly_chart(fig, use_container_width=True)
                        st.dataframe(rf_importance, use_container_width=True)
                
                with col2:
                    st.markdown("#### XGBoost")
                    xgb_importance = get_feature_importance(
                        models['XGBoost'], 
                        df_numeric.columns
                    )
                    if xgb_importance is not None:
                        fig = plot_feature_importance(xgb_importance, 'XGBoost Feature Importance')
                        st.plotly_chart(fig, use_container_width=True)
                        st.dataframe(xgb_importance, use_container_width=True)
                
                st.markdown("---")
                
                st.markdown("""
                #### Key Insights
                
                The most important features for predicting country development clusters are:
                
                1. **Child Mortality**: Strongest indicator of development level
                2. **Life Expectancy**: Reflects overall health and development
                3. **Income/GDPP**: Economic indicators crucial for classification
                4. **Health Spending**: Investment in healthcare infrastructure
                
                These features align with HELP International's focus areas.
                """)

elif page == "🎯 Recommendations":
    st.title("🎯 Aid Distribution Recommendations")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload data from the Home page first!")
    elif not st.session_state.clustering_done:
        st.warning("⚠️ Please complete clustering analysis first!")
    else:
        df = st.session_state.df
        kmeans_labels = st.session_state.kmeans_labels
        
        # Create analysis dataframe
        df_analysis = df.copy()
        df_analysis['Cluster'] = kmeans_labels
        
        tab1, tab2, tab3 = st.tabs(["📊 Cluster Analysis", "🚨 Priority Countries", "💰 Budget Allocation"])
        
        with tab1:
            st.markdown("### Cluster Characteristics")
            
            # Cluster statistics
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            cluster_stats = df_analysis.groupby('Cluster')[numeric_cols].mean()
            
            st.dataframe(cluster_stats.round(2), use_container_width=True)
            
            # Identify underdeveloped cluster
            cluster_dev_score = pd.DataFrame({
                'Cluster': cluster_stats.index,
                'Avg_Income': cluster_stats['income'],
                'Avg_GDPP': cluster_stats['gdpp'],
                'Avg_Child_Mort': cluster_stats['child_mort'],
                'Avg_Life_Expec': cluster_stats['life_expec']
            })
            
            cluster_dev_score['Development_Score'] = (
                cluster_dev_score['Avg_Income'] + 
                cluster_dev_score['Avg_GDPP'] - 
                cluster_dev_score['Avg_Child_Mort'] * 100 + 
                cluster_dev_score['Avg_Life_Expec'] * 100
            )
            
            cluster_dev_score = cluster_dev_score.sort_values('Development_Score')
            
            st.markdown("#### Development Ranking")
            st.dataframe(cluster_dev_score, use_container_width=True)
            
            most_underdeveloped = int(cluster_dev_score.iloc[0]['Cluster'])
            
            st.error(f"🚨 Most Underdeveloped Cluster: **Cluster {most_underdeveloped}**")
        
        with tab2:
            st.markdown("### Priority Countries for Aid")
            
            # Get underdeveloped countries
            underdeveloped_countries = df_analysis[
                df_analysis['Cluster'] == most_underdeveloped
            ].copy()
            
            # Calculate priority score
            underdeveloped_countries['Priority_Score'] = (
                (underdeveloped_countries['child_mort'] / underdeveloped_countries['child_mort'].max()) * config.PRIORITY_WEIGHTS['child_mort'] +
                (1 - underdeveloped_countries['life_expec'] / underdeveloped_countries['life_expec'].max()) * config.PRIORITY_WEIGHTS['life_expec'] +
                (1 - underdeveloped_countries['income'] / underdeveloped_countries['income'].max()) * config.PRIORITY_WEIGHTS['income'] +
                (underdeveloped_countries['total_fer'] / underdeveloped_countries['total_fer'].max()) * config.PRIORITY_WEIGHTS['total_fer'] +
                (1 - underdeveloped_countries['gdpp'] / underdeveloped_countries['gdpp'].max()) * config.PRIORITY_WEIGHTS['gdpp']
            )
            
            underdeveloped_countries = underdeveloped_countries.sort_values('Priority_Score', ascending=False)
            
            st.metric("Countries in Underdeveloped Cluster", len(underdeveloped_countries))
            
            # Top N slider
            top_n = st.slider("Number of top priority countries to display", 5, 20, 10)
            
            top_countries = underdeveloped_countries.head(top_n)[
                ['country', 'child_mort', 'life_expec', 'income', 'gdpp', 'Priority_Score']
            ].reset_index(drop=True)
            
            top_countries.index = top_countries.index + 1
            
            st.markdown(f"#### Top {top_n} Priority Countries")
            st.dataframe(top_countries, use_container_width=True)
            
            # Visualization
            fig = plot_priority_countries(underdeveloped_countries, top_n)
            st.plotly_chart(fig, use_container_width=True)
            
            # Save to session state
            st.session_state.top_countries = top_countries
            st.session_state.underdeveloped_countries = underdeveloped_countries
            
            st.markdown("---")
            
            # Detailed profiles
            st.markdown("#### Detailed Country Profiles")
            
            selected_country = st.selectbox(
                "Select country for detailed view",
                top_countries['country'].tolist()
            )
            
            country_data = underdeveloped_countries[
                underdeveloped_countries['country'] == selected_country
            ].iloc[0]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Child Mortality", f"{country_data['child_mort']:.1f} per 1000")
                st.metric("Life Expectancy", f"{country_data['life_expec']:.1f} years")
                st.metric("Total Fertility", f"{country_data['total_fer']:.2f}")
            
            with col2:
                st.metric("Income per Capita", f"${country_data['income']:.0f}")
                st.metric("GDP per Capita", f"${country_data['gdpp']:.0f}")
                st.metric("Health Spending", f"{country_data['health']:.2f}% GDP")
            
            with col3:
                st.metric("Exports", f"{country_data['exports']:.1f}% GDP")
                st.metric("Imports", f"{country_data['imports']:.1f}% GDP")
                st.metric("Inflation", f"{country_data['inflation']:.2f}%")
        
        with tab3:
            st.markdown("### Budget Allocation Strategy")
            
            st.info("💰 Total Budget Available: **$10,000,000**")
            
            if 'top_countries' in st.session_state:
                top_countries = st.session_state.top_countries
                
                allocation_strategy = st.radio(
                    "Select Allocation Strategy",
                    ["Equal Distribution", "Priority-Based", "Custom"]
                )
                
                if allocation_strategy == "Equal Distribution":
                    budget_per_country = 10_000_000 / len(top_countries)
                    allocations = [budget_per_country] * len(top_countries)
                
                elif allocation_strategy == "Priority-Based":
                    # Allocate based on priority score
                    total_priority = top_countries['Priority_Score'].sum()
                    allocations = (top_countries['Priority_Score'] / total_priority * 10_000_000).tolist()
                
                else:  # Custom
                    st.markdown("#### Custom Allocation")
                    allocations = []
                    for idx, country in enumerate(top_countries['country']):
                        amount = st.number_input(
                            f"{country}",
                            min_value=0,
                            max_value=10_000_000,
                            value=1_000_000,
                            step=100_000,
                            key=f"budget_{idx}"
                        )
                        allocations.append(amount)
                
                # Create allocation dataframe
                allocation_df = top_countries.copy()
                allocation_df['Allocated_Budget'] = allocations
                allocation_df['Budget_Formatted'] = allocation_df['Allocated_Budget'].apply(
                    lambda x: f"${x:,.0f}"
                )
                
                st.markdown("#### Budget Allocation Summary")
                st.dataframe(
                    allocation_df[['country', 'Priority_Score', 'Budget_Formatted']],
                    use_container_width=True
                )
                
                # Visualization
                fig = go.Figure(go.Pie(
                    labels=allocation_df['country'],
                    values=allocation_df['Allocated_Budget'],
                    hole=0.3
                ))
                fig.update_layout(title='Budget Distribution', height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Total Allocated", f"${sum(allocations):,.0f}")
                
                with col2:
                    remaining = 10_000_000 - sum(allocations)
                    st.metric("Remaining Budget", f"${remaining:,.0f}")
                
                st.session_state.allocation_df = allocation_df

elif page == "📥 Download Results":
    st.title("📥 Download Results")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please complete the analysis first!")
    else:
        st.markdown("### Available Downloads")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Cluster assignments
            if st.session_state.clustering_done:
                df_with_clusters = st.session_state.df.copy()
                df_with_clusters['Cluster'] = st.session_state.kmeans_labels
                
                csv_clusters = df_with_clusters.to_csv(index=False)
                st.download_button(
                    label="📊 Download Cluster Assignments (CSV)",
                    data=csv_clusters,
                    file_name="country_cluster_assignments.csv",
                    mime="text/csv"
                )
            
            # Priority countries
            if 'top_countries' in st.session_state:
                csv_priority = st.session_state.top_countries.to_csv(index=False)
                st.download_button(
                    label="🚨 Download Priority Countries (CSV)",
                    data=csv_priority,
                    file_name="priority_countries.csv",
                    mime="text/csv"
                )
        
        with col2:
            # Model results
            if st.session_state.models_trained:
                results_df = evaluate_models(
                    st.session_state.y_test,
                    st.session_state.predictions
                )
                csv_results = results_df.to_csv(index=False)
                st.download_button(
                    label="🤖 Download Model Results (CSV)",
                    data=csv_results,
                    file_name="model_performance.csv",
                    mime="text/csv"
                )
            
            # Budget allocation
            if 'allocation_df' in st.session_state:
                csv_allocation = st.session_state.allocation_df.to_csv(index=False)
                st.download_button(
                    label="💰 Download Budget Allocation (CSV)",
                    data=csv_allocation,
                    file_name="budget_allocation.csv",
                    mime="text/csv"
                )
        
        st.markdown("---")
        
        # Comprehensive report
        st.markdown("### 📄 Comprehensive Report")
        
        if st.button("Generate Full Report"):
            with st.spinner("Generating report..."):
                report_text = f"""
COUNTRY INTELLIGENCE SYSTEM - COMPREHENSIVE REPORT
HELP International
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

{'='*80}
EXECUTIVE SUMMARY
{'='*80}

Total Countries Analyzed: {len(st.session_state.df)}
Optimal Clusters: {optimal_k}
Countries Needing Aid: {len(st.session_state.underdeveloped_countries) if 'underdeveloped_countries' in st.session_state else 'N/A'}
Total Budget: $10,000,000

{'='*80}
TOP PRIORITY COUNTRIES
{'='*80}

{st.session_state.top_countries.to_string() if 'top_countries' in st.session_state else 'Please complete analysis'}

{'='*80}
MODEL PERFORMANCE
{'='*80}

{evaluate_models(st.session_state.y_test, st.session_state.predictions).to_string() if st.session_state.models_trained else 'Models not trained'}

{'='*80}
RECOMMENDATIONS
{'='*80}

1. Focus 50% of budget ($5M) on top 5 priority countries
2. Allocate remaining budget to next 10-15 countries
3. Primary focus areas:
   - Reduce child mortality
   - Improve healthcare infrastructure
   - Increase life expectancy
   - Support education programs

4. Monitor progress quarterly
5. Re-evaluate country clusters annually

{'='*80}
END OF REPORT
{'='*80}
                """
                
                st.download_button(
                    label="📥 Download Full Report (TXT)",
                    data=report_text,
                    file_name="HELP_International_Report.txt",
                    mime="text/plain"
                )

# Footer
# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            border-radius: 10px; margin-top: 30px;'>
    <div style='color: white;'>
        <h3 style='margin: 10px 0; color: white;'>🌍 Country Intelligence System</h3>
        <p style='margin: 5px 0; font-size: 14px;'>Powered by Machine Learning & AI</p>
        <hr style='border: 1px solid rgba(255,255,255,0.3); margin: 15px 0;'>
        <p style='margin: 10px 0; font-size: 16px; font-weight: bold;'>
            💻 Made by Aryan Dadhich | PIET
        </p>
        <p style='margin: 5px 0; font-size: 12px; opacity: 0.9;'>
            CEI Data Science & ML Intern
        </p>
    </div>
</div>
""", unsafe_allow_html=True)