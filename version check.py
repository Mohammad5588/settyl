import pickle
from sklearn.pipeline import Pipeline
from sklearn import __version__ as sklearn_version

# Load the saved model
with open('modeltrained.sav', 'rb') as f:
    model = pickle.load(f)

# Check the scikit-learn version
print("Current scikit-learn version:", sklearn_version)

# Re-save the model using the current version
with open('modeltrained_updated.sav', 'wb') as f:
    pickle.dump(model, f)
