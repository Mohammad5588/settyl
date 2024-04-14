import json
import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import numpy as np 
from fastapi import FastAPI
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
import pickle

with open('dataset.json', 'r') as f:
    data = json.load(f)
print(data[:5]) 


def clean_data(data1):
    external_clean = data1.replace("/","").strip()
    external_clean2 = re.sub(r'\(Vessel name\s*:\s*([^)]+)\)','',external_clean).strip()
    external_clean3 = external_clean2.lower()
    return external_clean3

def lower_data(data):
    internal_clean=data.lower()
    return internal_clean

for clean in data:
    clean['externalStatus'] = clean_data(clean['externalStatus'])
for clean in data:
    clean['internalStatus'] = lower_data(clean['internalStatus'])

    
externalStatus=[i['externalStatus'] for i in data]    
internalStatus=[i['internalStatus'] for i in data] 

    
df=pd.DataFrame({'external_status':externalStatus,'internal_status':internalStatus})

#try implement model with svc
label_encoder = LabelEncoder()
df['internal_status'] = label_encoder.fit_transform(df['internal_status'])

# Splitting X,Y
X_train, X_test, y_train, y_test = train_test_split(df['external_status'], df['internal_status'], test_size=0.2, random_state=42)

# pipeline for text classification
model = make_pipeline(
    CountVectorizer(),
    SVC(kernel='linear')
)

# Training phase
model.fit(X_train, y_train)

mlname='modeltrained.pkl'
f1=open(mlname,'wb')
pickle.dump(model,f1)
f.close()