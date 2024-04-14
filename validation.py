import json
import requests
from sklearn.metrics import accuracy_score

# Load the JSON validation data
with open('dataset.json', 'r') as f:
    validation_data = json.load(f)

# Send requests to the API and collect predictions
predictions = []
for item in validation_data:
    external_status = item['externalStatus']
    response = requests.post('http://127.0.0.1:8000/docs#/default/predict_internal_status_predict_internal_status_post', json={'external_status': external_status})
    if response.status_code == 200:
        predicted_internal_status = response.json().get('predicted_internal_status')
        predictions.append(predicted_internal_status)
    else:
        print(f"Error: Request failed for external status '{external_status}'")

# Extract actual internal status labels from the validation data
actual_internal_status = [item['internalStatus'] for item in validation_data]

# Calculate accuracy
accuracy = accuracy_score(actual_internal_status, predictions)
print(f"Accuracy: {accuracy}")

# Perform further analysis and evaluation as needed
