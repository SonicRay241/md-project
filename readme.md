# Obesity prediction model deployment with Streamlit and FastAPI

This project demonstrates a complete machine learning pipeline for predicting obesity levels using user-provided health metrics. The model is deployed using FastAPI as a backend API and Streamlit as a user-friendly frontend interface.

## 🚀 Overview

The goal of this project is to predict the likelihood of obesity based on personal and lifestyle data such as:
- Age
- Gender
- Height and Weight
- Family history with overweight
- Daily habits (physical activity, food consumption, etc.)

The pipeline includes:
- Data preprocessing and model training
- RESTful API deployment with FastAPI
- Interactive web interface with Streamlit
- Integration of frontend and backend for seamless predictions

## 🧰 Tech Stack
- Python 3.12
- scikit-learn / XGBoost for model training
- FastAPI – for serving the ML model as an API
- Streamlit – for frontend UI
- Uvicorn – for ASGI server to run FastAPI

## ✨ Features
- Real-time predictions from trained model
- REST API for integration with other services
- Clean and minimal frontend for end-users
- Easily extendable to include more features or models

## 📈 Model Performance
The model achieves high accuracy and F1-score on test data. Detailed evaluation metrics are available in the [model/training.ipynb](https://github.com/SonicRay241/md-project/blob/main/model/training.ipynb) notebook.

## 📄 License
This project is licensed under the [MIT License](https://github.com/SonicRay241/md-project/blob/main/LICENSE).