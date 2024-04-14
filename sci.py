import pickle

# Load the saved model
with open('modeltrained.pkl', 'rb') as f:
    model = pickle.load(f)

# Prepare some sample data (you can use any data similar to your training data)
sample_data = ["Port Out", "Vessel departure from first POL (Vessel name : TIAN FU HE)"]

# Make predictions using the loaded model
predictions = model.predict(sample_data)

# Print the predictions
for data, prediction in zip(sample_data, predictions):
    print(f"Data: {data}, Predicted internal status: {prediction}")

#192.168.0.109