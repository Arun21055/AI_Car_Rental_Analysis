# 🚗 Car Rental Feedback Analyzer (IBM watsonx.ai)

Uses IBM watsonx.ai's **FLAN-UL2** foundation model to analyse car rental customer comments:

1. **Satisfaction prediction:** was the customer satisfied (1) or not (0)?
2. **Business-area classification:** which area does the comment concern (pricing, staff attitude, etc.)?

Built during the **IBM SkillsBuild Generative AI internship (May–June 2025)**.

## Approach

**Few-shot prompting, with no model training.** Each task has a short instruction plus one worked example, followed by the customer comment. The model completes the answer.

```
comment ─▶ prompt (instruction + example + comment) ─▶ FLAN-UL2 (watsonx.ai) ─▶ label
```

| Task | Output | Max new tokens |
|---|---|---|
| Satisfaction | `0` or `1` | 10 |
| Business area | one of 6 categories | 15 |

Business areas: `Product: Functioning`, `Product: Pricing and Billing`, `Service: Accessibility`, `Service: Attitude`, `Service: Knowledge`, `Service: Orders/Contracts`.

## Tech stack

Python · pandas · scikit-learn · IBM watsonx.ai (Prompt Lab, FLAN-UL2) · IBM Watson Machine Learning SDK · IBM Cloud Object Storage

## What the script does

- Loads train/test CSVs from IBM Cloud Object Storage or from local files
- Sends each comment to FLAN-UL2 with a delay to avoid rate limits, and keeps going if a request fails
- Cleans the raw model output into a valid label
- Prints **accuracy and a classification report** when true labels exist
- Saves everything to `predictions.csv`

## Setup

```bash
pip install -r requirements.txt
```

Set credentials as environment variables (never commit keys):

```bash
export WML_API_KEY="your-watsonx-api-key"
export PROJECT_ID="your-watsonx-project-id"
```

**Data, option A: local CSV files**
```bash
export TEST_CSV=path/to/test_data.csv     # needs a Customer_Service column
```

**Data, option B: IBM Cloud Object Storage**
```bash
export COS_API_KEY=... COS_BUCKET=... COS_ENDPOINT=... COS_TEST_KEY=test_data.csv
```

## Run

```bash
python Car_Rental_Analysis.py
```

Expected columns: `Customer_Service` (comment text), optional `Satisfaction` (0/1) and `Business_Area` for evaluation.

## Limitations and future work

- Few-shot prompting with **one example per task**, so performance depends on prompt wording.
- The satisfaction prompt shows only a negative example, which may bias predictions. Adding a positive example is a quick improvement.
- The original dataset is in a private IBM Cloud bucket and is not included.
- No fine-tuning; a fine-tuned model or more examples could improve accuracy.
- The `ibm-watson-machine-learning` SDK is being replaced by `ibm-watsonx-ai`, so migrating is future work.

## Author

**P R Arun Kumar**, VIT-AP University
