import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from sklearn.metrics import accuracy_score
app = FastAPI()

pickle_in=open('modeltrained.pkl','rb')
model = pickle.load(pickle_in)

@app.get('/')
def index():
    return {'mess':'heelo'}

@app.get('/{name}')
def get_name(name:str):
    return{'mess':f'hello {name}'}


class status(BaseModel):
    externalStatus: str  # Change variable name to match the JSON structure

# Define the prediction endpoint
@app.post("/predict_internal_status")
async def predict_internal_status(external_status: status):
    # Make prediction using the loaded model
    prediction = model.predict([external_status.externalStatus])  # Access correct variable name
    predicted_internal_status = prediction[0]

    
    predicted_internal_status = int(predicted_internal_status)

    return {"predicted_internal_status": predicted_internal_status}




# Run the FastAPI app using uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)
