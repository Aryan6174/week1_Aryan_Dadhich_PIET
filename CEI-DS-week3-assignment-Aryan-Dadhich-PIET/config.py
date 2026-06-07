"""
Configuration settings for Country Intelligence System
"""

# App Settings
APP_TITLE = "🌍 Country Intelligence System"
APP_SUBTITLE = "AI-Powered Country Segmentation for HELP International"
APP_ICON = "🌍"

# Model Settings
RANDOM_STATE = 42
TEST_SIZE = 0.25
OPTIMAL_CLUSTERS = 3

# Feature Names
FEATURES = [
    'child_mort', 'exports', 'health', 'imports', 
    'income', 'inflation', 'life_expec', 'total_fer', 'gdpp'
]

# Feature Descriptions
FEATURE_DESCRIPTIONS = {
    'child_mort': 'Death of children under 5 years per 1000 live births',
    'exports': 'Exports of goods and services (% of GDP per capita)',
    'health': 'Total health spending (% of GDP per capita)',
    'imports': 'Imports of goods and services (% of GDP per capita)',
    'income': 'Net income per person',
    'inflation': 'Annual growth rate of Total GDP',
    'life_expec': 'Average life expectancy in years',
    'total_fer': 'Fertility rate (children per woman)',
    'gdpp': 'GDP per capita'
}

# Color Schemes
CLUSTER_COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1']
MODEL_COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']

# Priority Weights
PRIORITY_WEIGHTS = {
    'child_mort': 0.30,
    'life_expec': 0.25,
    'income': 0.25,
    'total_fer': 0.10,
    'gdpp': 0.10
}