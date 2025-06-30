!pip install datasets
!pip install scikit-learn
!pip install ibm-watson-machine-learning==1.0.312
import os, getpass
from pandas import read_csv
credentials = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "apikey": getpass.getpass("Please enter your WML api key (hit enter): ")
}
try:
    project_id = os.environ["PROJECT_ID"]
except KeyError:
    project_id = input("Please enter your project_id (hit enter): ")
project_id
import os, types
import pandas as pd
from ibm_boto3 import client
from botocore.client import Config
import ibm_boto3

def _iter_(self): return 0
cos_client = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='MFVLlm1YZ4zYK8XVZU4fjZ7gGNQ6d5V1v28MtLy9X90C',
    ibm_auth_endpoint="https://iam.cloud.ibm.com/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3.private.us-south.cloud-object-storage.appdomain.cloud'
)

bucket = 'handson-6gvhgxmhg'
object_key = 'train_data (1).csv'
body = cos_client.get_object(Bucket=bucket, Key=object_key)['Body']

if not hasattr(body, "_iter_"):
    body._iter_ = types.MethodType(_iter_, body)

train_data = pd.read_csv(body)
train_data.head(5)
import os, types
import pandas as pd
from botocore.client import Config
import ibm_boto3

def _iter_(self): return 0
cos_client = ibm_boto3.client(service_name='s3',
    ibm_api_key_id="MFVLlm1YZ4zYK8XVZU4fjZ7gGNQ6d5V1v28MtLy9X90C",
    ibm_auth_endpoint="https://iam.cloud.ibm.com/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url="https://s3.private.us-south.cloud-object-storage.appdomain.cloud"
)

bucket = 'handson-6gvhgxmhg'
object_key = 'test_data (1).csv'
body = cos_client.get_object(Bucket=bucket, Key=object_key)['Body']

if not hasattr(body, "_iter_"):
    body._iter_ = types.MethodType(_iter_, body)

test_data = pd.read_csv(body)
test_data.head(5)
train_data.shape
test_data.shape
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
model_id = ModelTypes.FLAN_UL2
satisfaction_instruction = """
Was customer satisfied?\n
comment: I have had a few recent rentals that have taken a very very long time, with no offer of apology.
In the most recent case, the agent subsequently offered me a car type on
an upgrade coupon and then told me it was no longer available because it had just be\n
satisfaction: 0\n\n
"""
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
parameters = {
    GenParams.MAX_NEW_TOKENS: 10
}
from ibm_watson_machine_learning.foundation_models import Model
model = Model(
    model_id=model_id,
    params=parameters,
    credentials=credentials,
    project_id=project_id
)
import time
results = []
comments = list(test_data.Customer_Service)
satisfaction = list(test_data.Satisfaction.astype(str))

for input_text in comments:
    prompt_text = " ".join([satisfaction_instruction, input_text])
    try:
        response = model.generate_text(prompt=prompt_text)
        results.append(response)
        time.sleep(0.6)  # Add delay to avoid 429 error
    except Exception as e:
        print(f"Error for input: {input_text[:30]}... => {e}")
        results.append("ERROR")
comments
satisfaction
results
business_area_instruction = """
Find the business area of the customer e-mail.
Choose business area from the following list:
'Product: Functioning', 'Product: Pricing and Billing', 'Service: Accessibility',
'Service: Attitude', 'Service: Knowledge', 'Service: Orders/Contracts'.

comment: I do not understand why I have to pay additional fee if vehicle is returned without a full tank.
business area: 'Product: Pricing and Billing'\n\n
"""
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

parameters = {
    GenParams.MAX_NEW_TOKENS: 15
}
from ibm_watson_machine_learning.foundation_models import Model

model = Model(
    model_id=model_id,
    params=parameters,
    credentials=credentials,
    project_id=project_id
)
import time
results = []
for input_text in comments:
    try:
        results.append(model.generate_text(prompt=" ".join([business_area_instruction, input_text])))
        time.sleep(0.6)  # wait to avoid hitting 2 req/sec limit
    except Exception as e:
        print("Error:", e)
        results.append("ERROR")

comments
area
results
