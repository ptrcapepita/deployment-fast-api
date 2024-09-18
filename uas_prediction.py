from builtins import float
import string
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import pickle as pkl
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
import uvicorn, logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictionRequest(BaseModel):
     age : int = Field(description="age of the client")
     job: object = Field(description="type of job (admin, blue-collar, entrepreneur, housemaid, management, retired, self-employed, services, student, technician, unemployed, unknown)")
     marital: object = Field(description="marital status (divorced, married, single, unknown)")
     education: object = Field(description="last education (basic.4y, basic.6y, basic.9y, high.school, illiterate, professional.course, university.degree, unknown)")
     default: object = Field(description="credit availability of the person (no, yes, unknown)")
     housing: object = Field(description="has house loan (no, yes, unknown)")
     loan: object = Field(description="has personal loan (no, yes, unknown)")
     contact: object = Field(description="contact communication type (cellular, telephone)")
     month: object = Field(description="last contact month of the year (jan, feb, mar, .., dec)")
     day_of_week: object = Field(description="last day of the week (mon, tue, wed, thu, fri)")
     duration: float = Field(description="last contact duration in seconds")
     campaign: int = Field(description="number of contacts performed during this campaign and for this client (incl. last contact)")
     pdays: int = Field(description="number of days that passed by after the client was last contacted from previous client (input 999 if client wasn't previously contacted)")
     previous: int = Field(description="number of contacts performed before this campaign and for this client")
     poutcome: object = Field(description="outcome of the previous marketing campaign (failure, nonexistent, success)")

     class Config:
        arbitrary_types_allowed = True  # Allow arbitrary types 

try:
    with open('uas_pickle_model.pkl', 'rb') as model_file:
        model = pkl.load(model_file)
    logger.info("Model loaded successfully")
    
    with open('uas_pickle_encode.pkl', 'rb') as le_file:
        encode = pkl.load(le_file)
    logger.info("Label encoders loaded successfully")
except Exception as e:
    logger.error(f"Error loading model or encoders: {e}")
    raise HTTPException(status_code=500, detail=f"Error loading model or encoders: {e}")

app = FastAPI()

@app.get("/")
def read_root():
       return {"message": "Welcome to the Machine Learning Model API"}

@app.post("/predict")
def predict(data: PredictionRequest):

    data = pd.DataFrame([data.dict().values()], columns=data.dict().keys())

    # Encode: Mapping
    mapping = {'unknown': 0, 'illiterate': 1, 'basic.4y': 2,
           'basic.6y': 3, 'basic.9y': 4, 'high.school': 5,
           'professional.course': 6, 'university.degree': 7}
    data['education'] = data['education'].map(mapping)

    mapping = {'mar': 3, 'apr': 4, 'may': 5,
           'jun': 6, 'jul': 7, 'aug': 8,
           'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
    data['month'] = data['month'].map(mapping)

    mapping = {'mon': 1, 'tue': 2, 'wed': 3,
           'thu': 4, 'fri': 5}
    data['day_of_week'] = data['day_of_week'].map(mapping)

    # Encode: OHE
    input_data = encode.transform(data)
    input_data = pd.DataFrame(input_data, columns=encode.get_feature_names_out())

    try:
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)
        
        print("Prediction:", prediction)
        print("Prediction Probability:", prediction_proba)
        
        return {"prediction": int(prediction[0]), "probability": prediction_proba[0].tolist()}
    
    except Exception as e:
        print("Error during prediction:", e)
        raise HTTPException(status_code=500, detail="Prediction error")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)