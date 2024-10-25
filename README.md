# Predictive Model Deployment with FastAPI

This project demonstrates the deployment of a predictive model using **FastAPI** to predict the likelihood of an individual subscribing to a service, based on a revenue bank dataset. It combines model building, tuning, and deployment into a production-ready API.

## Project Overview
The main steps in this project include:
- **Modeling and Evaluation**: Applied **Random Forest** and **Decision Tree** algorithms, evaluated model performance, and fine-tuned parameters to select the best-performing model.
- **Model Serialization**: Saved the tuned Random Forest model as a pickle file for use in the API.
- **API Deployment**: Used **FastAPI** to deploy the model, enabling users to make predictions on new data and assess subscription likelihood.

### Key Achievements:
- Achieved high prediction accuracy with a fine-tuned Random Forest model.
- Serialized the model and seamlessly integrated it into a FastAPI endpoint.
- Created a RESTful API that provides real-time predictions based on input features.
