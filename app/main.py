from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()
classifier = pipeline("sentiment-analysis")

class InputText(BaseModel):
    text: str

@app.get("/")
def health_check():
    return {"message": "API is running"}

@app.post("/predict")
def predict(data: InputText):
    result = classifier(data.text)
    return {"result": result}