"""
Car Rental Feedback Analyzer using IBM watsonx.ai (FLAN-UL2)

For each customer comment the script:
  1. predicts satisfaction (1 = satisfied, 0 = not satisfied), and
  2. classifies the business area of the comment.

It uses few-shot prompting of a foundation model (no model training) and
reports accuracy when ground-truth labels are available.

Credentials are read from environment variables. Never hardcode API keys.
    WML_API_KEY       watsonx.ai / Watson Machine Learning API key
    PROJECT_ID        watsonx.ai project id
    WATSONX_URL       (optional) default https://us-south.ml.cloud.ibm.com

Data (choose one):
    Local CSV files:  TRAIN_CSV / TEST_CSV  (paths)
    IBM Cloud Object Storage:  COS_API_KEY, COS_BUCKET, COS_ENDPOINT,
                               COS_TEST_KEY (and optionally COS_TRAIN_KEY)

The test CSV needs a `Customer_Service` column (comment text) and, for
evaluation, a `Satisfaction` column (0/1). A business-area label column
(`Business_Area`) is optional.
"""

import io
import os
import re
import time
import getpass

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

REQUEST_DELAY = 0.6  # seconds between requests, avoids rate-limit (429) errors

BUSINESS_AREAS = [
    "Product: Functioning",
    "Product: Pricing and Billing",
    "Service: Accessibility",
    "Service: Attitude",
    "Service: Knowledge",
    "Service: Orders/Contracts",
]

SATISFACTION_PROMPT = """
Was customer satisfied?

comment: I have had a few recent rentals that have taken a very very long time, with no offer of apology.
satisfaction: 0

comment: """

BUSINESS_AREA_PROMPT = """
Find the business area of the customer e-mail.
Choose business area from the following list:
'Product: Functioning', 'Product: Pricing and Billing', 'Service: Accessibility',
'Service: Attitude', 'Service: Knowledge', 'Service: Orders/Contracts'.

comment: I do not understand why I have to pay additional fee if vehicle is returned without a full tank.
business area: 'Product: Pricing and Billing'

comment: """


# ---------------------------------------------------------------- data loading
def _read_cos_csv(key: str) -> pd.DataFrame:
    import ibm_boto3
    from botocore.client import Config

    client = ibm_boto3.client(
        service_name="s3",
        ibm_api_key_id=os.environ["COS_API_KEY"],
        ibm_auth_endpoint="https://iam.cloud.ibm.com/oidc/token",
        config=Config(signature_version="oauth"),
        endpoint_url=os.environ["COS_ENDPOINT"],
    )
    body = client.get_object(Bucket=os.environ["COS_BUCKET"], Key=key)["Body"]
    return pd.read_csv(io.BytesIO(body.read()))


def load_data():
    """Load train/test data from local CSVs or IBM Cloud Object Storage."""
    if os.getenv("TEST_CSV"):
        train = pd.read_csv(os.environ["TRAIN_CSV"]) if os.getenv("TRAIN_CSV") else None
        test = pd.read_csv(os.environ["TEST_CSV"])
    else:
        train = _read_cos_csv(os.environ["COS_TRAIN_KEY"]) if os.getenv("COS_TRAIN_KEY") else None
        test = _read_cos_csv(os.environ["COS_TEST_KEY"])
    if train is not None:
        print("Train shape:", train.shape)
    print("Test shape:", test.shape)
    return train, test


# ------------------------------------------------------------------- the model
def build_model(max_new_tokens: int) -> Model:
    credentials = {
        "url": os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com"),
        "apikey": os.getenv("WML_API_KEY") or getpass.getpass("Enter your IBM WML API key: "),
    }
    project_id = os.getenv("PROJECT_ID") or input("Enter your project_id: ")
    return Model(
        model_id=ModelTypes.FLAN_UL2,
        params={GenParams.MAX_NEW_TOKENS: max_new_tokens},
        credentials=credentials,
        project_id=project_id,
    )


def generate_all(model: Model, prompt: str, comments):
    """Send every comment to the model; return a list of raw answers."""
    answers = []
    for text in comments:
        try:
            answers.append(model.generate_text(prompt=prompt + str(text)).strip())
        except Exception as exc:  # keep going if one request fails
            print(f"Error for '{str(text)[:30]}...': {exc}")
            answers.append("ERROR")
        time.sleep(REQUEST_DELAY)
    return answers


# ------------------------------------------------------------ output cleaning
def parse_satisfaction(raw: str) -> str:
    match = re.search(r"[01]", raw)
    return match.group(0) if match else "ERROR"


def parse_business_area(raw: str) -> str:
    cleaned = raw.strip().strip("'\"")
    for area in BUSINESS_AREAS:
        if area.lower() in cleaned.lower():
            return area
    return cleaned or "ERROR"


# ------------------------------------------------------------------------ main
def main():
    _, test = load_data()
    comments = list(test["Customer_Service"])

    # 1) Satisfaction prediction
    sat_model = build_model(max_new_tokens=10)
    sat_raw = generate_all(sat_model, SATISFACTION_PROMPT, comments)
    test["Predicted_Satisfaction"] = [parse_satisfaction(r) for r in sat_raw]

    # 2) Business-area classification
    area_model = build_model(max_new_tokens=15)
    area_raw = generate_all(area_model, BUSINESS_AREA_PROMPT, comments)
    test["Predicted_Business_Area"] = [parse_business_area(r) for r in area_raw]

    # Evaluation (only when true labels exist)
    if "Satisfaction" in test.columns:
        y_true = test["Satisfaction"].astype(int).astype(str)
        y_pred = test["Predicted_Satisfaction"]
        print("\nSatisfaction accuracy:", round(accuracy_score(y_true, y_pred), 4))
        print(classification_report(y_true, y_pred, zero_division=0))

    for col in ("Business_Area", "business_area", "Business Area"):
        if col in test.columns:
            acc = accuracy_score(test[col].astype(str).str.strip(),
                                 test["Predicted_Business_Area"])
            print(f"Business-area accuracy: {acc:.4f}")
            break

    test.to_csv("predictions.csv", index=False)
    print("\nSaved predictions.csv")
    print("Sample comment  :", comments[0])
    print("Predicted satisf:", test["Predicted_Satisfaction"].iloc[0])
    print("Predicted area  :", test["Predicted_Business_Area"].iloc[0])


if __name__ == "__main__":
    main()
