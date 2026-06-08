# 🌍 Country Intelligence System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

**AI-Powered Country Segmentation Platform for HELP International**

[Live Demo](https://cei-week3-assignment-aryan-dadhich-piet.streamlit.app/) • [Report Bug](https://github.com/Aryan6174/week1_Aryan_Dadhich_PIET/issues) • [Request Feature](https://github.com/Aryan6174/week1_Aryan_Dadhich_PIET/issues)

</div>

---

## 📋 Table of Contents

- [About The Project](#about-the-project)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Live Demo](#live-demo)
- [Installation](#installation)
- [Usage Guide](#usage-guide)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Results](#results)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

---

## 🎯 About The Project

The **Country Intelligence System** is a comprehensive machine learning web application designed to help **HELP International** (a humanitarian NGO) strategically allocate $10 million in aid to countries most in need.

### Problem Statement

HELP International has raised approximately $10 million and needs to make data-driven decisions about which countries require the most urgent humanitarian aid. The CEO needs insights on:

- Which countries are in the direst need of assistance
- How to categorize countries by development level
- Strategic allocation of limited resources for maximum impact

### Solution

This application uses **unsupervised learning (clustering)** to segment 167 countries based on socio-economic and health indicators, then applies **supervised learning (classification)** to build predictive models that can automatically categorize new countries.

---

## ✨ Key Features

### 🔍 **Exploratory Data Analysis**
- Interactive correlation heatmaps
- Distribution analysis with customizable visualizations
- Outlier detection and statistical summaries
- Missing value analysis and data quality checks

### 🎯 **Clustering Analysis**
- **K-Means Clustering** with elbow method optimization
- **DBSCAN** for density-based clustering
- Silhouette score analysis for cluster validation
- PCA visualization for dimensionality reduction

### 🤖 **Classification Models**
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier

### 📊 **Model Evaluation**
- Accuracy, Precision, Recall, F1-Score metrics
- Confusion matrices for all models
- Feature importance analysis
- Model comparison dashboard

### 🎯 **Aid Recommendations**
- Priority country ranking based on multiple indicators
- Budget allocation strategies (Equal, Priority-based, Custom)
- Detailed country profiles with development metrics
- Interactive budget distribution visualizations

### 📥 **Export Capabilities**
- Download cluster assignments (CSV)
- Export priority countries list
- Generate comprehensive reports
- Export budget allocation plans

---

## 🛠️ Tech Stack

### **Frontend & Framework**
- **Streamlit** - Interactive web application framework
- **Plotly** - Interactive visualizations
- **Matplotlib & Seaborn** - Statistical plotting

### **Data Processing**
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing

### **Machine Learning**
- **Scikit-learn** - ML models and preprocessing
- **XGBoost/Gradient Boosting** - Advanced ensemble methods

### **Deployment**
- **Streamlit Cloud** - Hosting platform
- **GitHub** - Version control

---

## 🚀 Live Demo

### 🌐 **Deployed Application**
**Access the live app here:** [Country Intelligence System](https://cei-week3-assignment-aryan-dadhich-piet.streamlit.app/)

### 📊 **Dataset**
- **Countries:** 167
- **Features:** 10 (socio-economic and health indicators)
- **Source:** Country development indicators dataset

### 📈 **Key Metrics**
- **Model Accuracy:** 95%+
- **Optimal Clusters:** 3
- **Priority Countries Identified:** 40+
- **Budget Available:** $10,000,000

---

## 💻 Installation

### Prerequisites

- Python 3.11 or higher
- pip package manager
- Git

### Local Setup

1. **Clone the repository**
```bash
git clone https://github.com/Aryan6174/week1_Aryan_Dadhich_PIET.git
cd CEI-DS-week3-assignment-Aryan-Dadhich-PIET
