# 🚗 Car Rental Feedback Analyzer using IBM watsonx.ai

This project uses IBM watsonx.ai's FLAN-UL2 foundation model to analyze customer service comments from a car rental company, predict satisfaction levels, and classify business areas.

---

## 📌 Project Overview

- Perform customer satisfaction prediction.
- Classify comments into business areas (e.g., Pricing, Attitude).
- Use IBM foundation models via watsonx and Python SDK.
- Retrieve data securely from IBM Cloud Object Storage.

---

## 🧰 Tools & Technologies Used

- IBM watsonx.ai & Prompt Lab
- Foundation Model: FLAN-UL2
- IBM Cloud Object Storage (COS)
- Python: `pandas`, `ibm_boto3`, `scikit-learn`
- IBM Watson Machine Learning SDK

---

## ⚙️ Full Project Code (Step-by-Step in One Block)

# 📦 Step 1: Install Required Packages
!pip install datasets
!pip install scikit-learn
!pip install ibm-watson-machine-learning==1.0.312

# 🔐 Step 2: Authentication and Imports
import os, getpass, types, time
import pandas as pd
from pandas import read_csv
from botocore.client import Config
import ibm_boto3

from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

# IBM Watson Credentials
credentials = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "apikey": getpass.getpass("Enter your IBM WML API key: ")
}

try:
    project_id = os.environ["PROJECT_ID"]
except KeyError:
    project_id = input("Enter your project_id: ")

# 🧾 Step 3: Load Data from IBM Cloud Object Storage (COS)
def _iter_(self): return 0

cos_client = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='YOUR_API_KEY',
    ibm_auth_endpoint="https://iam.cloud.ibm.com/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3.private.us-south.cloud-object-storage.appdomain.cloud'
)

bucket = 'handson-6gvhgxmhg'

# Load training data
object_key = 'train_data (1).csv'
body = cos_client.get_object(Bucket=bucket, Key=object_key)['Body']
if not hasattr(body, "_iter_"): body._iter_ = types.MethodType(_iter_, body)
train_data = pd.read_csv(body)

# Load test data
object_key = 'test_data (1).csv'
body = cos_client.get_object(Bucket=bucket, Key=object_key)['Body']
if not hasattr(body, "_iter_"): body._iter_ = types.MethodType(_iter_, body)
test_data = pd.read_csv(body)

print("Train shape:", train_data.shape)
print("Test shape:", test_data.shape)

# 🧠 Step 4: Load FLAN-UL2 Model for Satisfaction Prediction
model_id = ModelTypes.FLAN_UL2
parameters = { GenParams.MAX_NEW_TOKENS: 10 }

satisfaction_instruction = """
Was customer satisfied?

comment: I have had a few recent rentals that have taken a very very long time, with no offer of apology.
satisfaction: 0
"""

model = Model(
    model_id=model_id,
    params=parameters,
    credentials=credentials,
    project_id=project_id
)

# Step 5: Predict Customer Satisfaction
results_satisfaction = []
comments = list(test_data.Customer_Service)
satisfaction = list(test_data.Satisfaction.astype(str))

for input_text in comments:
    prompt_text = " ".join([satisfaction_instruction, input_text])
    try:
        response = model.generate_text(prompt=prompt_text)
        results_satisfaction.append(response)
        time.sleep(0.6)
    except Exception as e:
        print(f"Error: {e}")
        results_satisfaction.append("ERROR")

# 🏷️ Step 6: Predict Business Area Classification
business_area_instruction = """
Find the business area of the customer e-mail.
Choose from:
'Product: Functioning', 'Product: Pricing and Billing', 'Service: Accessibility',
'Service: Attitude', 'Service: Knowledge', 'Service: Orders/Contracts'.

comment: I do not understand why I have to pay additional fee if vehicle is returned without a full tank.
business area: 'Product: Pricing and Billing'
"""

parameters = { GenParams.MAX_NEW_TOKENS: 15 }

model = Model(
    model_id=model_id,
    params=parameters,
    credentials=credentials,
    project_id=project_id
)

results_business_area = []
for input_text in comments:
    try:
        prompt = " ".join([business_area_instruction, input_text])
        results_business_area.append(model.generate_text(prompt=prompt))
        time.sleep(0.6)
    except Exception as e:
        print("Error:", e)
        results_business_area.append("ERROR")

# Optional: View Final Outputs
print("Sample Comment:", comments[0])
print("Predicted Satisfaction:", results_satisfaction[0])
print("Predicted Business Area:", results_business_area[0])
