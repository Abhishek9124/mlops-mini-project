# MLOps Mini Project - Complete Analysis

**Project Owner:** Abhishek  
**License:** MIT  
**Date:** December 2025  
**Technology Stack:** Python, DVC, MLflow, DagsHub, Flask, scikit-learn, NLTK, pandas, numpy

---

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture & Pipeline](#architecture--pipeline)
3. [Detailed File-by-File Analysis](#detailed-file-by-file-analysis)
4. [Installation & Setup](#installation--setup)
5. [Execution Workflow](#execution-workflow)
6. [Technologies Used](#technologies-used)
7. [Commands Reference](#commands-reference)

---

## 🎯 Project Overview

This is an **MLOps (Machine Learning Operations) Mini Project** that implements a complete end-to-end machine learning pipeline for **Sentiment Analysis**. The project classifies text sentiments as either **Happy (1) or Sad (0)**.

### Key Features:
- ✅ Automated Data Ingestion & Preprocessing
- ✅ Feature Engineering (Bag of Words)
- ✅ Model Training (Logistic Regression)
- ✅ Model Evaluation & Metrics Tracking
- ✅ Model Registration & Versioning (MLflow + DagsHub)
- ✅ Flask Web Application for Predictions
- ✅ Complete Logging & Error Handling
- ✅ DVC Pipeline Orchestration

---

## 🏗️ Architecture & Pipeline

### Data Flow Diagram:
```
┌─────────────────────────────────────────────────────────────────┐
│                      DATA INGESTION                              │
│  - Fetch from Remote CSV (tweet_emotions.csv)                   │
│  - Split: 85% Train, 15% Test                                   │
│  - Output: data/raw/{train.csv, test.csv}                       │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   DATA PREPROCESSING                             │
│  - Remove URLs, numbers, punctuation                            │
│  - Lowercase text                                                │
│  - Remove stopwords                                              │
│  - Lemmatization                                                │
│  - Output: data/interim/{train_processed.csv, test_processed}   │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                  FEATURE ENGINEERING                             │
│  - Apply Bag of Words (CountVectorizer)                         │
│  - Max Features: 1000                                            │
│  - Output: data/processed/{train_bow.csv, test_bow.csv}        │
│  - Save Vectorizer: models/vectorizer.pkl                       │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL BUILDING                                │
│  - Algorithm: Logistic Regression                               │
│  - Parameters: C=1, solver='liblinear', penalty='l2'            │
│  - Output Model: models/model.pkl                               │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   MODEL EVALUATION                               │
│  - Metrics: Accuracy, Precision, Recall, AUC                   │
│  - MLflow Tracking (DagsHub Integration)                        │
│  - Output: reports/{metrics.json, experiment_info.json}         │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                  MODEL REGISTRATION                              │
│  - Register to MLflow Model Registry                            │
│  - Stage: Staging → Production                                  │
│  - DagsHub Tracking URI                                         │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                  FLASK WEB APPLICATION                           │
│  - REST API for predictions                                      │
│  - Load model from MLflow Registry                              │
│  - Real-time text preprocessing & prediction                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Detailed File-by-File Analysis

### 1. **setup.py** - Python Package Configuration

**Purpose:**  
Makes the project installable as a Python package so that the `src` module can be imported anywhere.

**File Location:** `setup.py`

**Key Content:**
```python
setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='A short description of the project.',
    author='Abhishek',
    license='MIT',
)
```

**Creation Steps:**
1. Define package metadata
2. Automatically find all sub-packages in the project
3. Configure version and license information

**Output:**
- Makes `src` importable via `pip install -e .`
- Enables relative imports throughout the project

**Tech Used:** `setuptools`, `Python packaging`

**Command to Create/Install:**
```bash
pip install -e .
```

---

### 2. **requirements.txt** - Project Dependencies

**Purpose:**  
Specifies all Python packages needed to run the entire project.

**File Location:** `requirements.txt`

**Key Dependencies (sample):**
```
pandas           # Data manipulation
numpy            # Numerical computing
scikit-learn     # Machine learning
nltk             # Natural Language Processing
mlflow           # MLflow tracking & registry
dagshub          # DagsHub integration
flask            # Web framework
pyyaml           # YAML parsing
boto3            # AWS integration
```

**Creation Steps:**
1. Identify all required packages
2. List with specific versions for reproducibility
3. Include both production and development dependencies

**Output:**
- 175+ packages installed
- Total dependencies across all sub-packages

**Tech Used:** `pip`, `Python package management`

**Commands:**
```bash
# Install all dependencies
pip install -r requirements.txt

# Generate requirements from current environment
pip freeze > requirements.txt

# Update specific package
pip install --upgrade scikit-learn
```

---

### 3. **params.yaml** - Hyperparameters Configuration

**Purpose:**  
Centralized configuration file for all ML hyperparameters and settings, allowing easy experimentation.

**File Location:** `params.yaml`

**Content:**
```yaml
data_ingestion:
  test_size: 0.15              # 15% test data, 85% training data

feature_engineering:
  max_features: 1000           # Maximum vocabulary size for CountVectorizer
```

**Creation Steps:**
1. Define data split ratio for train-test split
2. Configure feature extraction parameters
3. Keep all tunable parameters in one YAML file

**Output:**
- Easy parameter tracking for experiments
- Simple modification for hyperparameter tuning
- DVC integration for parameter versioning

**Tech Used:** `YAML`, `DVC tracking`

**Usage Commands:**
```bash
# View parameters
cat params.yaml

# Modify and rerun pipeline
# Edit params.yaml, then run: dvc repro
```

---

### 4. **dvc.yaml** - DVC Pipeline Definition

**Purpose:**  
Defines the complete ML pipeline stages as a Directed Acyclic Graph (DAG), enabling reproducible automation.

**File Location:** `dvc.yaml`

**Content Structure:**
```yaml
stages:
  data_ingestion:
    cmd: python src/data/data_ingestion.py
    deps: [src/data/data_ingestion.py]
    params: [data_ingestion.test_size]
    outs: [data/raw]
    
  data_preprocessing:
    cmd: python src/data/data_preprocessing.py
    deps: [data/raw, src/data/data_preprocessing.py]
    outs: [data/interim]
    
  feature_engineering:
    cmd: python src/features/feature_engineering.py
    deps: [data/interim, src/features/feature_engineering.py]
    params: [feature_engineering.max_features]
    outs: [data/processed, models/vectorizer.pkl]
    
  model_building:
    cmd: python src/model/model_building.py
    deps: [data/processed, src/model/model_building.py]
    outs: [models/model.pkl]
    
  model_evaluation:
    cmd: python src/model/model_evaluation.py
    deps: [models/model.pkl, src/model/model_evaluation.py]
    metrics: [reports/metrics.json]
    outs: [reports/experiment_info.json]
    
  model_registration:
    cmd: python src/model/register_model.py
    deps: [reports/experiment_info.json, src/model/register_model.py]
```

**Creation Steps:**
1. Identify each stage in the ML pipeline
2. Define command to execute
3. List dependencies (input files/scripts)
4. List outputs (generated artifacts)
5. Link parameters from params.yaml

**Output:**
- DAG visualization of pipeline
- Automatic stage execution in correct order
- Caching of unchanged stages

**Tech Used:** `DVC (Data Version Control)`, `YAML`

**Commands:**
```bash
# Initialize DVC
dvc init

# Run entire pipeline
dvc repro

# Run specific stage
dvc repro model_building

# Visualize pipeline DAG
dvc dag

# Visualize with browser
dvc plots diff
```

---

### 5. **src/data/data_ingestion.py** - Data Loading & Splitting

**Purpose:**  
Fetches raw data from remote source, preprocesses it, and splits into train/test sets.

**File Location:** `src/data/data_ingestion.py`

**Key Functions:**

#### `load_params(params_path: str) -> dict`
- **Purpose:** Load hyperparameters from YAML
- **Input:** `params.yaml`
- **Output:** Dictionary with parameters
- **Error Handling:** FileNotFoundError, YAMLError

#### `load_data(data_url: str) -> pd.DataFrame`
- **Purpose:** Fetch data from remote URL
- **Input:** URL to CSV file
- **Output:** Pandas DataFrame
- **Source:** `https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv`
- **Size:** Original dataset with multiple sentiment classes

#### `preprocess_data(df: pd.DataFrame) -> pd.DataFrame`
- **Purpose:** Initial data cleaning
- **Steps:**
  1. Drop 'tweet_id' column
  2. Filter only 'happiness' and 'sadness' sentiments
  3. Encode: happiness→1, sadness→0
- **Output:** Binary classification dataset

#### `save_data(train_data, test_data, data_path) -> None`
- **Purpose:** Save splits to disk
- **Output Files:**
  - `data/raw/train.csv`
  - `data/raw/test.csv`

**Execution Flow:**
```
load_params('params.yaml')
    ↓
load_data(remote_url)
    ↓
preprocess_data(df)
    ↓
train_test_split(test_size=0.15, random_state=42)
    ↓
save_data(train_data, test_data)
```

**Output:**
- **File:** `data/raw/train.csv` (~1000+ rows, 2 columns)
- **File:** `data/raw/test.csv` (~150+ rows, 2 columns)
- **Columns:** ['content', 'sentiment']
- **Logging:** errors.log

**Tech Used:** `pandas`, `scikit-learn`, `PyYAML`, `logging`

**Command to Run Directly:**
```bash
python src/data/data_ingestion.py
```

---

### 6. **src/data/data_preprocessing.py** - Text Normalization

**Purpose:**  
Applies advanced NLP preprocessing to clean and normalize text data.

**File Location:** `src/data/data_preprocessing.py`

**Key Functions:**

#### Text Cleaning Functions:

1. **`lower_case(text: str) -> str`**
   - Converts text to lowercase
   - Example: "HELLO World" → "hello world"

2. **`removing_urls(text: str) -> str`**
   - Removes HTTP/HTTPS URLs
   - Regex: `https?://\S+|www\.\S+`

3. **`removing_numbers(text: str) -> str`**
   - Removes all digits
   - Example: "Hello123" → "Hello"

4. **`removing_punctuations(text: str) -> str`**
   - Removes punctuation marks
   - Handles special Unicode characters
   - Example: "Hello, World!" → "Hello World"

5. **`remove_stop_words(text: str) -> str`**
   - Removes common English words (the, is, at, etc.)
   - Uses NLTK stopwords list
   - Reduces noise in data

6. **`lemmatization(text: str) -> str`**
   - Converts words to root form
   - Example: "running" → "run", "better" → "good"
   - Uses WordNetLemmatizer

7. **`normalize_text(df: pd.DataFrame) -> pd.DataFrame`**
   - Applies all preprocessing steps in sequence
   - Works on 'content' column

**Processing Pipeline:**
```
Raw Text
  ↓ lower_case()
  ↓ removing_urls()
  ↓ removing_numbers()
  ↓ removing_punctuations()
  ↓ remove_stop_words()
  ↓ lemmatization()
Cleaned Text
```

**Example Transformation:**
```
Input:  "I LOVE this! Visit https://example.com for more 123 info!!!"
  ↓
Output: "love visit info"
```

**Output:**
- **File:** `data/interim/train_processed.csv`
- **File:** `data/interim/test_processed.csv`
- **Logging:** transformation_errors.log

**Tech Used:** `nltk`, `regex`, `pandas`, `logging`

**Command to Run Directly:**
```bash
python src/data/data_preprocessing.py
```

---

### 7. **src/features/feature_engineering.py** - Vectorization

**Purpose:**  
Converts text data into numerical features using Bag of Words model.

**File Location:** `src/features/feature_engineering.py`

**Key Functions:**

#### `apply_bow(train_data, test_data, max_features) -> tuple`
- **Purpose:** Apply CountVectorizer to text data
- **Algorithm:** Bag of Words (BoW)
- **Steps:**
  1. Initialize CountVectorizer with max_features=1000
  2. Fit on training data text
  3. Transform train and test data
  4. Convert sparse matrix to dense DataFrame
  5. Append sentiment labels as final column
  
**Input:**
- Preprocessed text from `data/interim/`
- max_features: 1000 (vocabulary size limit)

**Output:**
- Dense matrices with shape (n_samples, 1000)
- Last column: sentiment labels

**Example Output Structure:**
```
     0     1     2  ...  999  label
0    1     0     2  ...   0     1
1    0     1     0  ...   1     0
...
```

**Vectorizer Persistence:**
- **File:** `models/vectorizer.pkl`
- **Purpose:** Used during inference to transform new text
- **Size:** ~500KB (vocabulary + statistics)

**Output Files:**
- `data/processed/train_bow.csv` (1001 columns: 1000 features + 1 label)
- `data/processed/test_bow.csv`

**Tech Used:** `scikit-learn CountVectorizer`, `pickle`, `pandas`, `logging`

**Command to Run Directly:**
```bash
python src/features/feature_engineering.py
```

---

### 8. **src/model/model_building.py** - Model Training

**Purpose:**  
Trains a Logistic Regression classifier on BoW features.

**File Location:** `src/model/model_building.py`

**Key Functions:**

#### `train_model(X_train, y_train) -> LogisticRegression`
- **Algorithm:** Logistic Regression (Linear classification)
- **Parameters:**
  - `C=1`: Regularization strength (inverse)
  - `solver='liblinear'`: Efficient for small datasets
  - `penalty='l2'`: Ridge regularization
  
**Training Process:**
```
Load BoW Features
  ↓
Split into X (features) and y (labels)
  ↓
Initialize LogisticRegression(C=1, solver='liblinear', penalty='l2')
  ↓
Fit model: clf.fit(X_train, y_train)
  ↓
Save model: pickle.dump(clf, 'models/model.pkl')
```

**Input:**
- File: `data/processed/train_bow.csv`
- Shape: (n_train_samples, 1001) where last column is label

**Output Model:**
- **File:** `models/model.pkl`
- **Model Type:** scikit-learn LogisticRegression object
- **File Size:** ~50KB
- **Logging:** model_building_errors.log

**What's Saved:**
- Model coefficients (1000 weights)
- Intercept (bias term)
- Classes (0, 1)
- Hyperparameters

**Tech Used:** `scikit-learn`, `pickle`, `pandas`, `numpy`, `logging`

**Command to Run Directly:**
```bash
python src/model/model_building.py
```

---

### 9. **src/model/model_evaluation.py** - Model Assessment

**Purpose:**  
Evaluates model performance on test set and logs metrics to MLflow/DagsHub.

**File Location:** `src/model/model_evaluation.py`

**Key Functions:**

#### `evaluate_model(clf, X_test, y_test) -> dict`
- **Purpose:** Calculate performance metrics
- **Metrics Calculated:**

1. **Accuracy** = (TP + TN) / (TP + TN + FP + FN)
   - Overall correctness
   - Example: 0.7778 (77.78%)

2. **Precision** = TP / (TP + FP)
   - When predicting Happy, how often correct?
   - Example: 0.7707 (77.07%)

3. **Recall** = TP / (TP + FN)
   - Of all Happy instances, how many found?
   - Example: 0.7788 (77.88%)

4. **AUC (Area Under ROC Curve)**
   - Trade-off between True Positive Rate and False Positive Rate
   - Example: 0.8640 (86.40%)
   - Better discrimination ability

**Metric Results:**
```json
{
    "accuracy": 0.7777777777777778,
    "precision": 0.7707253886010362,
    "recall": 0.7787958115183246,
    "auc": 0.863995662306966
}
```

#### `save_metrics(metrics, file_path)`
- **Output:** `reports/metrics.json`

#### MLflow Integration:
- **Tracking URI:** `https://dagshub.com/Abhishek9124/mlops-mini-project.mlflow`
- **Purpose:** Log metrics to remote experiment tracker
- **Requires:** DAGSHUB_PAT environment variable
- **Output:** `reports/experiment_info.json` (run_id, model_path)

**Execution Flow:**
```
Load model.pkl
  ↓
Load test data
  ↓
Make predictions
  ↓
Calculate metrics
  ↓
Log to MLflow
  ↓
Save metrics.json & experiment_info.json
```

**Output Files:**
- `reports/metrics.json` - Model evaluation metrics
- `reports/experiment_info.json` - MLflow run information
- `model_evaluation_errors.log` - Error logging

**Tech Used:** `scikit-learn metrics`, `MLflow`, `DagsHub`, `pickle`, `logging`

**Command to Run Directly:**
```bash
python src/model/model_evaluation.py
```

**Environment Setup:**
```bash
export DAGSHUB_PAT=your_dagshub_token
```

---

### 10. **src/model/register_model.py** - Model Registry

**Purpose:**  
Registers the trained model to MLflow Model Registry for version control and deployment.

**File Location:** `src/model/register_model.py`

**Key Functions:**

#### `register_model(model_name, model_info)`
- **Purpose:** Register model version to MLflow
- **Input:** model_info JSON with run_id and model_path
- **Output:** Registered model version in MLflow
- **Staging:** Transitions model to "Staging" stage

**Model Registry Stages:**
```
Development/Staging/Production
     ↓
  Trained Model
     ↓
  Register Version (v1, v2, v3, etc.)
     ↓
  Promote through Stages
     ↓
  Production Deployment
```

**Registration Flow:**
```
Load experiment_info.json (contains run_id)
  ↓
Construct model_uri: runs:/{run_id}/{model_path}
  ↓
Register to MLflow Registry
  ↓
Get model version number
  ↓
Transition to Staging stage
  ↓
Ready for promotion to Production
```

**Output:**
- Registered model in DagsHub MLflow registry
- Model URI: `models:/my_model/Staging`
- Version tracking enabled
- Logging: model_registration_errors.log

**Tech Used:** `MLflow`, `DagsHub`, `logging`

**Command to Run Directly:**
```bash
python src/model/register_model.py
```

---

### 11. **flask_app/app.py** - Web Application

**Purpose:**  
Provides REST API endpoints for real-time sentiment prediction via Flask web server.

**File Location:** `flask_app/app.py`

**Key Features:**

#### Text Preprocessing Functions:
All preprocessing functions mirrored from data preprocessing:
- `lower_case()` - Lowercase conversion
- `removing_urls()` - URL removal
- `removing_numbers()` - Number removal
- `removing_punctuations()` - Punctuation removal
- `remove_stop_words()` - Stopwords removal
- `lemmatization()` - Lemmatization
- `normalize_text()` - Complete pipeline

#### Model & Vectorizer Loading:
```python
# Load from MLflow Registry
MODEL_URI = "models:/mlops_model/Staging"
model = mlflow.pyfunc.load_model(MODEL_URI)

# Load saved vectorizer
vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))
```

#### Flask Routes:

**1. `GET /` - Home Page**
- Renders HTML form
- Input: Textarea for user text
- Output: HTML template with prediction

**2. `POST /predict` - Prediction Endpoint**
- Input: Text from form (raw user input)
- Process:
  1. Extract text from form
  2. Validate input (not empty)
  3. Normalize text using preprocessing pipeline
  4. Vectorize using loaded vectorizer
  5. Make prediction using MLflow model
  6. Return result (Happy/Sad)
- Output: Rendered template with prediction

**Example Usage:**
```
Input Text:  "I love this amazing weather!"
Processing:
  → Normalize: "love amazing weather"
  → Vectorize: [0, 1, 0, ..., 1, 0, 0] (1000 features)
  → Predict: 1 (Happy)
Output: "Happy"
```

**Prediction Mapping:**
- Prediction = 1 → "Happy"
- Prediction = 0 → "Sad"

**Configuration:**
- **Port:** 5000
- **Debug:** True
- **Host:** 0.0.0.0 (accessible from any network)

**Requirements:**
- DAGSHUB_PAT environment variable
- models/vectorizer.pkl file
- MLflow model registry access
- Flask template: flask_app/templates/index.html

**Output:**
- Web interface for interactive predictions
- REST API endpoint for programmatic access

**Tech Used:** `Flask`, `MLflow`, `pickle`, `NLTK`, `pandas`, `scikit-learn`

**Commands to Run:**
```bash
# Run Flask development server
python flask_app/app.py

# In another terminal, test endpoint
curl -X POST http://localhost:5000/predict -d "text=I love this"
```

---

### 12. **flask_app/templates/index.html** - Web Interface

**Purpose:**  
HTML template for user-friendly sentiment prediction interface.

**File Location:** `flask_app/templates/index.html`

**HTML Structure:**
```html
<!DOCTYPE html>
<html>
<head>
    <title>Sentiment Analysis</title>
</head>
<body style="background-color: brown">
    <h1>Sentiment Analysis</h1>
    <form action="/predict" method="POST">
        <label>Write text:</label><br>
        <textarea name="text" rows="10" cols="40"></textarea><br>
        <input type="submit" value="Predict">
    </form>
    
    {% if result is not none %}
        {% if result == 1 %}
            <h2>Happy</h2>
        {% else %}
            <h2>Sad</h2>
        {% endif %}
    {% endif %}
</body>
</html>
```

**Features:**
- Simple, intuitive interface
- Large textarea for user input (10 rows × 40 columns)
- Submit button for prediction
- Jinja2 templating for dynamic results
- Brown background styling

**User Flow:**
1. User visits http://localhost:5000/
2. Sees form with textarea
3. Types sentiment text
4. Clicks "Predict"
5. Form POSTs to /predict endpoint
6. Flask processes & returns updated template
7. Result displays as "Happy" or "Sad"

**Tech Used:** `HTML`, `Jinja2`, `CSS (inline)`

---

### 13. **flask_app/preprocessing_utility.py** - Utility Module

**Purpose:**  
Reusable text preprocessing functions (modular design).

**File Location:** `flask_app/preprocessing_utility.py`

**Functions:**
- `lemmatization(text)` - Convert to root words
- `remove_stop_words(text)` - Remove common words
- `removing_numbers(text)` - Remove digits
- `lower_case(text)` - Lowercase conversion
- `removing_punctuations(text)` - Remove punctuation
- `removing_urls(text)` - Remove URLs
- `remove_small_sentences(df)` - Filter short sentences

**Purpose of Modular Design:**
- Avoid code duplication
- Reusable across multiple files
- Easier maintenance and testing

**Tech Used:** `NLTK`, `regex`, `pandas`

---

### 14. **tests/test_model.py** - Model Unit Tests

**Purpose:**  
Automated testing for model loading, inference, and performance validation.

**File Location:** `tests/test_model.py`

**Test Cases:**

#### `TestModelLoading` Class:

1. **`setUpClass()`**
   - Load DagsHub credentials from environment
   - Connect to MLflow tracking URI
   - Load latest model from registry
   - Load vectorizer from disk
   - Load holdout test data

2. **`test_model_loaded_properly()`**
   - Verifies model object is not None
   - Ensures model loaded successfully from registry

3. **`test_model_signature()`**
   - Tests model input/output shape
   - Validates expected feature count (1000)

4. **Other potential tests:**
   - `test_predictions_binary` - Output is 0 or 1
   - `test_model_accuracy_threshold` - Meets minimum performance
   - `test_inference_speed` - Prediction latency acceptable

**Test Execution:**
```bash
python -m pytest tests/test_model.py -v
python -m unittest tests.test_model -v
```

**Tech Used:** `unittest`, `MLflow`, `pandas`, `scikit-learn`

---

### 15. **tests/test_flask_app.py** - Flask Application Tests

**Purpose:**  
Test Flask application endpoints and functionality.

**File Location:** `tests/test_flask_app.py`

**Test Cases:**

#### `FlaskAppTests` Class:

1. **`setUpClass()`**
   - Initialize Flask test client
   - Load Flask app for testing

2. **`test_home_page()`**
   - HTTP GET request to "/"
   - Verify status code = 200
   - Check for HTML title "Sentiment Analysis"

3. **`test_predict_page()`**
   - HTTP POST to "/predict" with test text
   - Verify status code = 200
   - Check response contains "Happy" or "Sad"

**Test Execution:**
```bash
python -m pytest tests/test_flask_app.py -v
python -m unittest tests.test_flask_app -v
```

**Tech Used:** `unittest`, `Flask test client`

---

### 16. **Makefile** - Build Automation

**Purpose:**  
Automate common project tasks and commands.

**File Location:** `Makefile`

**Available Commands:**

```makefile
# Install dependencies
make requirements

# Create dataset
make data

# Clean compiled files
make clean

# Lint source code
make lint

# Sync data to AWS S3
make sync_data_to_s3

# Sync data from AWS S3
make sync_data_from_s3
```

**Tech Used:** `GNU Make`, `bash`

**Commands to Use:**
```bash
make requirements      # Install all dependencies
make clean            # Remove .pyc and __pycache__
make lint             # Run flake8 on src/
```

---

### 17. **test_environment.py** - Environment Verification

**Purpose:**  
Verify Python environment meets project requirements.

**File Location:** `test_environment.py`

**Verification:**
- Checks if Python 3 is installed
- Validates Python version compatibility

**Command:**
```bash
python test_environment.py
# Output: ">>> Development environment passes all tests!"
```

**Tech Used:** `Python sys module`

---

### 18. **LICENSE** - MIT License

**Purpose:**  
Legal license for open-source distribution.

**File Location:** `LICENSE`

**Details:**
- Type: MIT License
- Author: Abhishek
- Year: 2025
- Permissions: Commercial use, modification, distribution
- Requirements: License and copyright notice
- Limitations: Liability, warranty

---

### 19. **README.md** - Original Project Documentation

**Purpose:**  
Standard project overview and structure documentation.

**File Location:** `README.md`

**Contents:**
- Project organization diagram
- File structure explanation
- Links to standard data science template

---

## 📊 Data Files Structure

### Raw Data: `data/raw/`
```
train.csv          ~1000+ rows, 2 columns: [content, sentiment]
test.csv           ~150+ rows, 2 columns: [content, sentiment]
```

### Interim Data: `data/interim/`
```
train_processed.csv    Cleaned text from train.csv
test_processed.csv     Cleaned text from test.csv
```

### Processed Data: `data/processed/`
```
train_bow.csv          1001 columns: 1000 features + 1 label
test_bow.csv           1001 columns: 1000 features + 1 label
```

---

## 📦 Model & Artifact Files

### Models: `models/`
```
model.pkl              Trained Logistic Regression model (~50KB)
vectorizer.pkl         CountVectorizer for feature transformation (~500KB)
```

### Reports: `reports/`
```
metrics.json           Model evaluation metrics:
                       - accuracy: 0.7778
                       - precision: 0.7707
                       - recall: 0.7788
                       - auc: 0.8640

experiment_info.json   MLflow experiment information:
                       - run_id: unique identifier
                       - model_path: location in registry
```

---

## 🔧 Installation & Setup

### Step 1: Clone Repository
```bash
git clone https://github.com/Abhishek9124/mlops-mini-project.git
cd mlops-mini-project
```

### Step 2: Create Virtual Environment
```bash
# Using venv
python -m venv mlops-env
source mlops-env/Scripts/activate  # Windows
# or
source mlops-env/bin/activate      # Linux/Mac
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Install Project in Development Mode
```bash
pip install -e .
```

### Step 5: Configure DagsHub (Optional but Recommended)
```bash
# Set environment variable
export DAGSHUB_PAT=your_dagshub_token  # Linux/Mac
setx DAGSHUB_PAT your_dagshub_token    # Windows
```

### Step 6: Initialize DVC (Data Version Control)
```bash
dvc init
dvc remote add -d myremote s3://bucket-name  # Optional
```

---

## 🚀 Execution Workflow

### Option 1: Run Full Pipeline with DVC

```bash
# Execute entire pipeline
dvc repro

# This automatically runs stages in correct order:
# data_ingestion → data_preprocessing → feature_engineering 
# → model_building → model_evaluation → model_registration
```

### Option 2: Run Individual Stages

```bash
# Data ingestion
python src/data/data_ingestion.py

# Data preprocessing
python src/data/data_preprocessing.py

# Feature engineering
python src/features/feature_engineering.py

# Model building
python src/model/model_building.py

# Model evaluation
python src/model/model_evaluation.py

# Model registration
python src/model/register_model.py
```

### Option 3: Run Flask Web Application

```bash
# Start Flask server
python flask_app/app.py

# Access in browser
# http://localhost:5000

# For production
gunicorn -w 4 flask_app.app:app
```

### Option 4: Run Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_model.py -v
python -m pytest tests/test_flask_app.py -v

# Run with coverage
pytest --cov=src tests/
```

---

## 🛠️ Technologies Used

### Core ML Libraries
| Technology | Purpose | Version |
|-----------|---------|---------|
| **scikit-learn** | ML algorithms (Logistic Regression) | 1.x |
| **pandas** | Data manipulation & analysis | 2.x |
| **numpy** | Numerical computing | 1.26+ |
| **NLTK** | NLP preprocessing | 3.x |

### MLOps & Tracking
| Technology | Purpose | Version |
|-----------|---------|---------|
| **DVC** | Data version control & pipeline | 3.x |
| **MLflow** | Experiment tracking & model registry | 2.x |
| **DagsHub** | Hosted MLflow & Git integration | - |

### Web Framework
| Technology | Purpose | Version |
|-----------|---------|---------|
| **Flask** | Web application framework | 2.x |
| **Jinja2** | HTML templating | 3.x |

### Utilities
| Technology | Purpose | Version |
|-----------|---------|---------|
| **PyYAML** | YAML file parsing | 6.x |
| **pickle** | Model serialization | Built-in |
| **boto3** | AWS integration | 1.26+ |
| **celery** | Task queue | 5.x |
| **pytest** | Unit testing | 7.x |

### Infrastructure
| Technology | Purpose |
|-----------|---------|
| **Python** | Programming language (3.10+) |
| **pip** | Package manager |
| **Make** | Build automation |
| **Git** | Version control |
| **GitHub** | Repository hosting |

---

## � DVC (Data Version Control) - Foundation & Commands

### Why DVC? The Core Problem It Solves

**The Problem:**
```
Traditional Git + Data Science = Issues:
├── ❌ Git cannot handle large files (>100MB)
├── ❌ No versioning for datasets and models
├── ❌ No automated pipeline execution
├── ❌ Manual experiment tracking
├── ❌ Difficult reproducibility
└── ❌ No dependency management for data stages
```

**DVC Solution:**
```
DVC = Git for Data + Machine Learning Pipelines

Features:
├── ✅ Version large files & directories
├── ✅ Track data, models, and metrics
├── ✅ Define & execute ML pipelines (dvc.yaml)
├── ✅ Reproduce exact results every time
├── ✅ Track experiments & parameters
└── ✅ Integrate with Git workflows
```

### DVC Architecture

```
┌────────────────────────────────────────┐
│     Git Repository (Code + Config)     │
│  - Source files (.py, .yaml)           │
│  - dvc.yaml (pipeline definition)      │
│  - params.yaml (hyperparameters)       │
│  - .dvc files (pointers to data)       │
└────────────────────────────────────────┘
            │
            │ .dvcignore, .gitignore
            ▼
┌────────────────────────────────────────┐
│   Local Working Directory              │
│  - data/ (actual datasets)             │
│  - models/ (trained models)            │
│  - reports/ (metrics & outputs)        │
└────────────────────────────────────────┘
            │
            │ dvc push/pull
            ▼
┌────────────────────────────────────────┐
│   Remote Storage (S3, Azure, etc)      │
│  - Large files cache & versioning      │
│  - Data backup & sharing               │
│  - Team collaboration                  │
└────────────────────────────────────────┘
```

### DVC Installation & Initialization

#### Step 1: Install DVC
```bash
# Using pip
pip install dvc

# Using conda
conda install -c conda-forge dvc

# Verify installation
dvc version
```

#### Step 2: Initialize DVC in Repository
```bash
# Initialize DVC in existing Git repo
git init
dvc init

# Creates:
# ├── .dvc/ (DVC configuration directory)
# │   ├── config (DVC configuration)
# │   ├── .gitignore
# │   └── plots/
# └── .dvcignore (like .gitignore for DVC)
```

#### Step 3: Configure Remote Storage
```bash
# AWS S3 (most common)
dvc remote add -d myremote s3://my-bucket/dvc-storage
dvc remote modify myremote --local access_key_id <YOUR_ACCESS_KEY>
dvc remote modify myremote --local secret_access_key <YOUR_SECRET_KEY>

# Azure Blob Storage
dvc remote add -d myremote azure://my-container/path
dvc remote modify myremote --local connection_string <CONNECTION_STRING>

# Google Cloud Storage
dvc remote add -d myremote gs://my-bucket/dvc-storage

# Local directory (for testing)
dvc remote add -d myremote /tmp/dvc-storage

# List configured remotes
dvc remote list
```

#### Step 4: Configure with DagsHub (Recommended for MLOps)
```bash
# DagsHub combines Git hosting + DVC storage + MLflow tracking
dvc remote add -d myremote 'https://dagshub.com/<USERNAME>/<REPO_NAME>.dvc'
dvc remote modify -d myremote auth basic
dvc remote modify myremote --local username <YOUR_USERNAME>
dvc remote modify myremote --local password <YOUR_TOKEN>

# Example:
dvc remote add -d myremote 'https://dagshub.com/Abhishek9124/mlops-mini-project.dvc'
```

### DVC Pipeline Fundamentals

#### Understanding dvc.yaml Structure

```yaml
stages:
  stage_name:                              # Unique stage identifier
    cmd: python script/path.py             # Command to execute
    
    deps:                                  # Input dependencies
      - input_file.csv
      - script/path.py
    
    params:                                # Parameter dependencies (from params.yaml)
      - parameter_key.subkey
    
    outs:                                  # Output artifacts (tracked by DVC)
      - output_file.csv
    
    metrics:                               # Metrics to track (JSON/YAML/CSV)
      - metrics.json:
          cache: false                     # Don't cache metrics file
    
    plots:                                 # Plot definitions (optional)
      - plots_data.csv:
          x: epoch
          y: loss
    
    deps-cache: true                       # Cache dependencies
```

### Complete DVC Commands Reference

#### **1. Pipeline Execution**

**`dvc repro` - Reproduce Entire Pipeline**
```bash
# Run full pipeline from start to finish
dvc repro

# Run pipeline with specific target
dvc repro models/model.pkl

# Rerun only changed stages
dvc repro --downstream data_preprocessing

# Force rerun all stages
dvc repro --force

# Dry run (show what would execute)
dvc repro --dry
```

**Why:** Ensures reproducible experiments. Only reruns changed stages, saving time.

**Output:**
```
Running stage 'data_ingestion'... Done
Running stage 'data_preprocessing'... Done
Running stage 'feature_engineering'... Done
Running stage 'model_building'... Done
Running stage 'model_evaluation'... Done
Running stage 'model_registration'... Done
```

---

#### **2. Pipeline Visualization**

**`dvc dag` - Display Pipeline DAG**
```bash
# Show pipeline structure
dvc dag

# Output:
# ┌─────────────────┐
# │ data_ingestion  │
# └────────┬────────┘
#          │
#    ┌─────▼──────────┐
#    │ data_preproc... │
#    └─────┬──────────┘
#          │
#    ┌─────▼─────────────┐
#    │ feature_enginee... │
#    └─────┬─────────────┘
#          │
#    ┌─────▼──────────┐
#    │ model_building │
#    └─────┬──────────┘
#          │
#    ┌─────▼───────────────┐
#    │ model_evaluation    │
#    └─────┬───────────────┘
#          │
#    ┌─────▼────────────────┐
#    │ model_registration   │
#    └──────────────────────┘
```

**Why:** Visual understanding of dependencies and pipeline flow.

---

#### **3. Data Tracking**

**`dvc add` - Track Large Files/Directories**
```bash
# Add entire data directory
dvc add data/raw

# Creates:
# ├── data/raw (directory with actual data)
# └── data/raw.dvc (pointer file, added to Git)

# Add individual file
dvc add models/model.pkl

# Add multiple items
dvc add data/raw data/processed models/
```

**Why:** Version control for large files without Git storage limits.

**`dvc status` - Check Pipeline & Data Status**
```bash
# Show if data/outputs are up-to-date
dvc status

# Output examples:
# Changed deps:
#   data/raw/train.csv
# 
# New outputs:
#   data/interim/train_processed.csv

# Everything up-to-date
# (no changes)
```

**Why:** Quick verification that all outputs match current inputs.

---

#### **4. Remote Storage Operations**

**`dvc push` - Upload to Remote**
```bash
# Push all data to configured remote
dvc push

# Push specific file
dvc push data/raw.dvc

# Push to specific remote
dvc push -r myremote

# Show what would be pushed (dry run)
dvc push --dry

# Output:
# Preparing to push data to 's3://my-bucket/dvc-storage'...
# 3 file(s) and 2 dir(s) uploaded successfully
```

**Why:** Backup data, share with team, store in persistent location.

---

**`dvc pull` - Download from Remote**
```bash
# Pull all data from remote
dvc pull

# Pull specific file
dvc pull data/raw.dvc

# Pull from specific remote
dvc pull -r myremote

# Only download files that don't exist locally
dvc pull --relink

# Output:
# Fetching data from 's3://my-bucket/dvc-storage'...
# 3 file(s) and 2 dir(s) downloaded successfully
```

**Why:** Get data on new machine, restore after git clone, collaborate.

---

**`dvc fetch` - Download Without Checking Out**
```bash
# Fetch updates to local cache
dvc fetch

# Don't update working directory
dvc fetch --remote myremote

# Useful for CI/CD to prepare without modifying files
```

---

#### **5. Parameter & Metrics Tracking**

**`dvc params diff` - Compare Parameter Changes**
```bash
# Show parameter differences from last commit
dvc params diff

# Compare between branches
dvc params diff main --all

# Output:
# Path         Param                  Old    New
# params.yaml  data_ingestion.test... 0.15   0.20
# params.yaml  feature_engineer...    1000   2000
```

**Why:** Track hyperparameter experiments and changes.

---

**`dvc metrics diff` - Compare Model Metrics**
```bash
# Compare metrics from last commit
dvc metrics diff

# Compare between branches
dvc metrics diff main

# Output:
# Path                Old      New      Change
# reports/metrics.json
#   accuracy          0.75     0.78     +0.03
#   precision         0.74     0.77     +0.03
#   recall            0.76     0.78     +0.02
#   auc               0.84     0.86     +0.02
```

**Why:** Track model performance improvements across experiments.

---

#### **6. Experiment Tracking**

**`dvc exp run` - Run Experiment**
```bash
# Run with parameters from params.yaml
dvc exp run

# Run with modified parameters
dvc exp run -S feature_engineering.max_features=2000

# Run multiple experiments
dvc exp run -q -n exp1 -S data_ingestion.test_size=0.1
dvc exp run -q -n exp2 -S data_ingestion.test_size=0.2
dvc exp run -q -n exp3 -S data_ingestion.test_size=0.3
```

**Why:** Quick hyperparameter tuning and comparison.

---

**`dvc exp show` - Display Experiment Results**
```bash
# Show all experiments
dvc exp show

# Show only main branches
dvc exp show --all-branches

# Show with specific metrics/params
dvc exp show --only-changed

# Output (table format):
# ┌──────────────┬──────────┬──────────┬───────────┬──────────┐
# │ Experiment   │ Accuracy │ Loss     │ Max Feats │ Test Size│
# ├──────────────┼──────────┼──────────┼───────────┼──────────┤
# │ main         │ 0.7778   │ 0.45     │ 1000      │ 0.15     │
# │ exp1         │ 0.7850   │ 0.42     │ 1500      │ 0.15     │
# │ exp2         │ 0.7920   │ 0.40     │ 2000      │ 0.15     │
# └──────────────┴──────────┴──────────┴───────────┴──────────┘
```

**Why:** Compare experiment results side-by-side.

---

**`dvc exp compare` - Detailed Experiment Comparison**
```bash
# Compare specific experiments
dvc exp compare exp1 exp2 exp3

# Compare with baseline
dvc exp compare --baseline main exp1

# Show all parameters and metrics
dvc exp compare --all
```

---

#### **7. Caching & Optimization**

**`dvc cache` - Manage Cache**
```bash
# Show cache directory location
dvc cache dir

# Show cache statistics
dvc cache dir --show-size

# Validate cache integrity
dvc cache verify

# Remove unused cache
dvc cache prune

# Clear all cache
dvc cache remove --not-in-remote
```

**Why:** Manage disk space, verify data integrity, optimize storage.

---

#### **8. Configuration**

**`dvc config` - Configure DVC Settings**
```bash
# View all configuration
dvc config --list

# Set default remote
dvc config core.remote myremote

# Set autostage (automatic .dvc file staging)
dvc config core.autostage true

# Set analytics
dvc config core.analytics false

# Configure DagsHub credentials
dvc config -l core.remote.dagshub
dvc remote modify dagshub --local username <username>
dvc remote modify dagshub --local password <token>

# Check config file locations
cat .dvc/config           # Local config
cat .dvc/config.local     # Secrets (not in Git)
```

---

#### **9. Advanced Commands**

**`dvc import` - Import from Another Repository**
```bash
# Import versioned data from another repo
dvc import https://github.com/user/repo data/dataset.csv

# Creates:
# ├── data/dataset.csv (downloaded)
# └── data/dataset.csv.dvc (reference to source)
```

---

**`dvc run` - Define Pipeline Stage Manually**
```bash
# Define a stage (alternative to dvc.yaml)
dvc run -n training \
    -d data/train.csv \
    -o models/model.pkl \
    -M reports/metrics.json \
    python src/model/model_building.py

# Creates stage in dvc.yaml
```

---

**`dvc plots` - Visualize Metrics**
```bash
# Show all plots
dvc plots show

# Interactive plots in browser
dvc plots diff main

# Custom plot configuration
dvc plots templates
```

---

### DVC Workflow Summary

#### Complete Workflow Example:

```bash
# 1. Initialize DVC
git init
dvc init

# 2. Configure remote storage
dvc remote add -d myremote s3://my-bucket/path

# 3. Create dvc.yaml with pipeline stages
# (Already created in this project)

# 4. Add large files to DVC
dvc add data/raw

# 5. Commit to Git (only .dvc files, not actual data)
git add dvc.yaml .dvc .gitignore
git commit -m "Add DVC pipeline"

# 6. Push data to remote
dvc push

# 7. Run pipeline
dvc repro

# 8. Check results
dvc metrics show
dvc params show

# 9. Track experiments
dvc exp run -S feature_engineering.max_features=2000
dvc exp show

# 10. Commit changes
git add dvc.lock
git commit -m "Update pipeline results"

# 11. Share with team
git push
dvc push
```

---

## 🚀 GitHub Actions CI/CD - Automated ML Workflows

### Why GitHub Actions for MLOps?

**The Problem Without CI/CD:**
```
Manual ML Workflow = Problems:
├── ❌ Manual testing on local machine
├── ❌ No automated model validation
├── ❌ Inconsistent results across environments
├── ❌ Risk of pushing broken code
├── ❌ Slow feedback on changes
├── ❌ Manual deployment process
└── ❌ No automated monitoring
```

**GitHub Actions Solution:**
```
CI/CD Automation Benefits:
├── ✅ Automated testing on every push
├── ✅ Consistent environment for all runs
├── ✅ Immediate feedback on code quality
├── ✅ Automatic model retraining
├── ✅ Automated deployment
├── ✅ Historical run tracking
└── ✅ Scheduled pipeline execution
```

### MLOps Foundation Architecture

```
┌─────────────────────────────────────────────────────────┐
│           GitHub Repository (Code)                       │
│  ├─ Python source files                                 │
│  ├─ dvc.yaml (DVC pipeline)                            │
│  ├─ params.yaml (hyperparameters)                      │
│  └─ .github/workflows/ (GitHub Actions)                │
└─────────┬───────────────────────────────────────────────┘
          │
          │ Push Trigger
          ▼
┌─────────────────────────────────────────────────────────┐
│    GitHub Actions Runners (CI/CD)                       │
│                                                          │
│  Job 1: Code Quality                                    │
│  ├─ Lint (flake8, pylint)                             │
│  ├─ Type checking (mypy)                              │
│  └─ Security scanning                                  │
│                                                          │
│  Job 2: Unit Tests                                      │
│  ├─ pytest tests/test_model.py                        │
│  ├─ pytest tests/test_flask_app.py                    │
│  └─ Coverage report                                     │
│                                                          │
│  Job 3: ML Pipeline                                     │
│  ├─ dvc repro (retrain model)                         │
│  ├─ dvc metrics show (track results)                  │
│  └─ dvc push (save artifacts)                         │
│                                                          │
│  Job 4: Deployment                                      │
│  ├─ Build Docker image                                │
│  ├─ Deploy to staging                                 │
│  └─ Run smoke tests                                    │
└─────────┬───────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────┐
│      Production Environment                             │
│  ├─ DVC remote storage (data + models)                 │
│  ├─ MLflow tracking server (DagsHub)                   │
│  └─ Flask application (deployed)                       │
└─────────────────────────────────────────────────────────┘
```

### GitHub Actions Fundamentals

#### File Structure:
```
.github/
├── workflows/
│   ├── ci-pipeline.yml          (Main CI/CD workflow)
│   ├── ml-pipeline.yml          (ML retraining)
│   ├── tests.yml                (Test execution)
│   ├── deploy.yml               (Deployment)
│   └── scheduled-training.yml   (Cron-based training)
└── workflows-config/
    └── env-variables.yml        (Shared config)
```

#### Workflow Syntax Basics:
```yaml
name: Workflow Name

on:                                    # Trigger conditions
  push:                               # Run on push
    branches: [main, develop]
  pull_request:                       # Run on PR
    branches: [main]
  schedule:                           # Run on schedule
    - cron: '0 0 * * *'              # Daily at midnight

jobs:
  job-name:                          # Job identifier
    runs-on: ubuntu-latest           # Runner environment
    
    steps:                           # Sequence of commands
      - name: Step description
        run: command
        
      - name: Another step
        uses: action/from/marketplace
```

### Complete CI/CD Workflow for MLOps Project

#### **1. Code Quality & Testing Workflow**

Create: `.github/workflows/ci-pipeline.yml`

```yaml
name: CI Pipeline - Code Quality & Tests

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  code-quality:
    name: Code Quality Checks
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
          cache: 'pip'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install flake8 pylint black mypy pytest-cov
      
      - name: Lint with flake8
        run: |
          # Stop on syntax errors
          flake8 src --count --select=E9,F63,F7,F82 --show-source --statistics
          # Warn on style issues
          flake8 src --count --exit-zero --max-complexity=10 --max-line-length=100
      
      - name: Format check with Black
        run: black --check src/ tests/
      
      - name: Type checking with mypy
        run: mypy src --ignore-missing-imports
        continue-on-error: true
      
      - name: Security scan
        run: |
          pip install bandit
          bandit -r src/
        continue-on-error: true

  unit-tests:
    name: Unit Tests
    runs-on: ubuntu-latest
    needs: code-quality
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
          cache: 'pip'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      
      - name: Run tests
        run: |
          pip install pytest pytest-cov
          pytest tests/ -v --cov=src --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml
          fail_ci_if_error: true

  integration-tests:
    name: Integration Tests
    runs-on: ubuntu-latest
    needs: unit-tests
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
          cache: 'pip'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      
      - name: Test Flask endpoint
        run: |
          python -m pytest tests/test_flask_app.py -v
      
      - name: Test model loading
        run: |
          python -m pytest tests/test_model.py -v
```

**Why Each Step:**
- **Checkout**: Get repository code
- **Setup Python**: Prepare environment
- **Flake8**: Check for style violations
- **Black**: Ensure code formatting
- **mypy**: Catch type errors
- **bandit**: Find security issues
- **pytest**: Run unit & integration tests
- **codecov**: Track test coverage

---

#### **2. ML Pipeline Retraining Workflow**

Create: `.github/workflows/ml-pipeline.yml`

```yaml
name: ML Pipeline - Train & Evaluate

on:
  push:
    branches: [main]
    paths:
      - 'src/**'
      - 'params.yaml'
      - 'dvc.yaml'
  workflow_dispatch:  # Manual trigger
  schedule:
    - cron: '0 3 * * 0'  # Weekly at 3 AM UTC on Sunday

jobs:
  train-model:
    name: Train ML Model
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v3
        with:
          fetch-depth: 0  # Full history for DVC
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install dvc[s3]  # For S3 support
      
      - name: Configure DVC
        run: |
          dvc remote add -d myremote ${{ secrets.DVC_REMOTE }}
          dvc remote modify myremote --local access_key_id ${{ secrets.AWS_ACCESS_KEY }}
          dvc remote modify myremote --local secret_access_key ${{ secrets.AWS_SECRET_KEY }}
      
      - name: Pull data from DVC
        run: |
          dvc pull
      
      - name: Run ML Pipeline
        run: |
          dvc repro
      
      - name: Show metrics
        run: |
          dvc metrics show
          cat reports/metrics.json
      
      - name: Push updated models & data
        run: |
          dvc push
      
      - name: Commit metrics changes
        run: |
          git config --local user.email "action@github.com"
          git config --local user.name "GitHub Action"
          git add dvc.lock reports/metrics.json
          git commit -m "Update metrics from GitHub Actions" || true
          git push
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Configure MLflow
        run: |
          export DAGSHUB_PAT=${{ secrets.DAGSHUB_PAT }}
          export MLFLOW_TRACKING_URI=${{ secrets.MLFLOW_TRACKING_URI }}
      
      - name: Register model
        run: |
          python src/model/register_model.py
        env:
          DAGSHUB_PAT: ${{ secrets.DAGSHUB_PAT }}
      
      - name: Create model report
        if: always()
        run: |
          echo "# Model Training Report" > model_report.md
          echo "Date: $(date)" >> model_report.md
          echo "## Metrics" >> model_report.md
          cat reports/metrics.json >> model_report.md
          echo "## Parameters" >> model_report.md
          cat params.yaml >> model_report.md
      
      - name: Comment PR with metrics
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v6
        with:
          script: |
            const fs = require('fs');
            const metrics = JSON.parse(fs.readFileSync('reports/metrics.json'));
            
            const comment = `## 📊 Model Metrics
            - **Accuracy**: ${(metrics.accuracy * 100).toFixed(2)}%
            - **Precision**: ${(metrics.precision * 100).toFixed(2)}%
            - **Recall**: ${(metrics.recall * 100).toFixed(2)}%
            - **AUC**: ${(metrics.auc * 100).toFixed(2)}%`;
            
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: comment
            });
```

**Key Features:**
- **Triggers**: On code changes, manual run, or schedule
- **DVC Integration**: Pull data → Run pipeline → Push artifacts
- **MLflow Tracking**: Register models to production registry
- **Git Commits**: Auto-commit updated metrics
- **PR Comments**: Post metrics on pull requests

---

#### **3. Docker Build & Deployment Workflow**

Create: `.github/workflows/deploy.yml`

```yaml
name: Build & Deploy

on:
  push:
    branches: [main]
    tags: ['v*']
  workflow_dispatch:

jobs:
  build-and-push:
    name: Build Docker Image
    runs-on: ubuntu-latest
    
    outputs:
      image-tag: ${{ steps.meta.outputs.tags }}
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v3
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2
      
      - name: Login to Docker Hub
        uses: docker/login-action@v2
        with:
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}
      
      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v4
        with:
          images: ${{ secrets.DOCKER_REGISTRY }}/mlops-app
          tags: |
            type=ref,event=branch
            type=semver,pattern={{version}}
            type=sha
      
      - name: Build and push
        uses: docker/build-push-action@v4
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  deploy-staging:
    name: Deploy to Staging
    runs-on: ubuntu-latest
    needs: build-and-push
    
    steps:
      - name: Deploy to AWS
        run: |
          # Deploy to ECS, Kubernetes, or other platform
          aws ecs update-service \
            --cluster staging \
            --service mlops-api \
            --force-new-deployment
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_KEY }}
          AWS_REGION: us-east-1
      
      - name: Run smoke tests
        run: |
          sleep 30  # Wait for deployment
          curl -f http://staging-api.example.com/health || exit 1

  deploy-production:
    name: Deploy to Production
    runs-on: ubuntu-latest
    needs: deploy-staging
    if: startsWith(github.ref, 'refs/tags/v')
    
    steps:
      - name: Deploy to production
        run: |
          aws ecs update-service \
            --cluster production \
            --service mlops-api \
            --force-new-deployment
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_KEY }}
          AWS_REGION: us-east-1
      
      - name: Health check
        run: |
          curl -f http://api.example.com/health || exit 1
      
      - name: Notify Slack
        uses: slackapi/slack-github-action@v1
        with:
          payload: |
            {
              "text": "✅ Production deployment successful!",
              "blocks": [
                {
                  "type": "section",
                  "text": {
                    "type": "mrkdwn",
                    "text": "*Production Deployment Successful*\n*Tag*: ${{ github.ref }}"
                  }
                }
              ]
            }
```

---

#### **4. Scheduled Retraining Workflow**

Create: `.github/workflows/scheduled-training.yml`

```yaml
name: Scheduled Model Retraining

on:
  schedule:
    # Train model daily at 2 AM UTC
    - cron: '0 2 * * *'
    # Weekly comprehensive retraining at 3 AM Sunday
    - cron: '0 3 * * 0'
  workflow_dispatch:

jobs:
  retrain:
    name: Retrain Model Daily
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install dvc[s3]
      
      - name: Configure DVC & MLflow
        run: |
          dvc remote add -d myremote ${{ secrets.DVC_REMOTE }}
          dvc remote modify myremote --local access_key_id ${{ secrets.AWS_ACCESS_KEY }}
          dvc remote modify myremote --local secret_access_key ${{ secrets.AWS_SECRET_KEY }}
        env:
          DAGSHUB_PAT: ${{ secrets.DAGSHUB_PAT }}
      
      - name: Pull latest data
        run: dvc pull
      
      - name: Retrain pipeline
        run: dvc repro
        env:
          DAGSHUB_PAT: ${{ secrets.DAGSHUB_PAT }}
      
      - name: Check metrics improvement
        id: metrics
        run: |
          python -c "
          import json
          with open('reports/metrics.json') as f:
            metrics = json.load(f)
          print(f'accuracy={metrics[\"accuracy\"]}')
          print(f'::set-output name=accuracy::{metrics[\"accuracy\"]}')" 
      
      - name: Notify if metrics degraded
        if: failure()
        uses: actions/github-script@v6
        with:
          script: |
            github.rest.issues.createComment({
              owner: context.repo.owner,
              repo: context.repo.repo,
              issue_number: 1,
              body: '⚠️ Model retraining failed or metrics degraded'
            });
```

---

### GitHub Secrets Configuration

**Required Secrets to Add:**

1. Go to: Settings → Secrets and variables → Actions
2. Add secrets:

```bash
# AWS S3 for DVC storage
AWS_ACCESS_KEY          # AWS access key ID
AWS_SECRET_KEY          # AWS secret access key

# DagsHub for MLflow & DVC
DAGSHUB_PAT            # DagsHub personal access token

# Docker registry
DOCKER_USERNAME        # Docker Hub username
DOCKER_PASSWORD        # Docker Hub access token

# AWS deployment
AWS_REGION             # AWS region (e.g., us-east-1)

# MLflow tracking
MLFLOW_TRACKING_URI    # MLflow server URI

# Slack notifications (optional)
SLACK_WEBHOOK          # Slack webhook for notifications
```

**Command to Add Secrets (from CLI):**
```bash
# Using GitHub CLI
gh secret set AWS_ACCESS_KEY --body "$AWS_KEY"
gh secret set AWS_SECRET_KEY --body "$AWS_SECRET"
gh secret set DAGSHUB_PAT --body "$DAGSHUB_TOKEN"
```

---

### MLOps Foundation Complete Setup

#### **Dockerfile** for Containerization

Create: `Dockerfile`

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/ ./src/
COPY flask_app/ ./flask_app/
COPY models/ ./models/
COPY dvc.yaml .
COPY params.yaml .

# Expose Flask port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

# Run Flask app
CMD ["python", "-m", "flask_app.app"]
```

#### **.gitignore** Configuration

Create/Update: `.gitignore`

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.egg-info/
.Python
env/
venv/

# DVC (data & models)
/data/raw/
/data/interim/
/data/processed/
/models/*.pkl
*.dvc
dvc.lock

# MLflow
mlruns/
.mlflow

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# Logs
*.log
errors.log

# Environment
.env
.env.local
.env.*.local

# Cache
.dvc/cache/
.pytest_cache/
.coverage
htmlcov/
```

#### **.dvcignore** Configuration

Create: `.dvcignore`

```
# Ignore small test files
test*.csv
sample*.csv

# Ignore intermediate files
*.tmp
*.bak

# Ignore IDE files
.vscode/
.idea/

# Ignore logs
*.log
```

---

### CI/CD Workflow Triggers Explained

#### **Trigger Types:**

```yaml
on:
  # 1. Push to specific branches
  push:
    branches: [main, develop]
    paths: ['src/**', 'params.yaml']  # Only if these files change
  
  # 2. Pull Request events
  pull_request:
    branches: [main]
  
  # 3. Manual trigger
  workflow_dispatch:
    inputs:
      environment:
        description: 'Deployment environment'
        required: true
        default: 'staging'
  
  # 4. Scheduled runs (cron syntax)
  schedule:
    - cron: '0 2 * * *'    # Daily at 2 AM
    - cron: '0 3 * * 0'    # Weekly Sunday 3 AM
  
  # 5. Release published
  release:
    types: [published]
  
  # 6. Repository dispatch (API trigger)
  repository_dispatch:
    types: [trigger-training]
```

---

### Running Workflows Manually

```bash
# List workflows
gh workflow list

# Run workflow manually
gh workflow run ml-pipeline.yml

# Check workflow status
gh run list --workflow=ml-pipeline.yml

# View workflow logs
gh run view RUN_ID --log

# Cancel workflow
gh run cancel RUN_ID
```

---

### Best Practices for CI/CD

```
✅ BEST PRACTICES:

1. Environment Separation
   ├─ Secrets: Use GitHub secrets
   ├─ Staging: Test before production
   └─ Monitoring: Alert on failures

2. Fast Feedback
   ├─ Parallel jobs
   ├─ Caching dependencies
   └─ Early exit on failures

3. Data Management
   ├─ DVC for large files
   ├─ Remote storage (S3, etc)
   └─ Versioning every artifact

4. Model Validation
   ├─ Unit tests
   ├─ Integration tests
   ├─ Performance benchmarks
   └─ A/B testing

5. Monitoring & Alerts
   ├─ Slack/Email notifications
   ├─ Metrics tracking
   ├─ Performance regression alerts
   └─ Deployment tracking
```

---

## �📋 Commands Reference

### Setup Commands
```bash
# Virtual environment
python -m venv mlops-env
source mlops-env/Scripts/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Verify environment
python test_environment.py
```

### Pipeline Execution
```bash
# Full pipeline
dvc repro

# Specific stage
dvc repro model_evaluation

# View pipeline DAG
dvc dag
dvc plots diff
```

### Model Management
```bash
# Train & evaluate
python src/data/data_ingestion.py
python src/data/data_preprocessing.py
python src/features/feature_engineering.py
python src/model/model_building.py
python src/model/model_evaluation.py
python src/model/register_model.py
```

### Web Application
```bash
# Run Flask server (development)
python flask_app/app.py

# Run with Gunicorn (production)
gunicorn -w 4 flask_app.app:app

# Test endpoint
curl -X POST http://localhost:5000/predict -d "text=I love this"
```

### Testing
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest --cov=src tests/ --cov-report=html

# Run specific test
pytest tests/test_model.py::TestModelLoading::test_model_loaded_properly -v
```

### Code Quality
```bash
# Lint code
flake8 src

# Clean compiled files
make clean

# Remove all generated artifacts
rm -rf data/raw data/interim data/processed
rm -rf models/*.pkl
rm -rf reports/*.json
```

### Data Management
```bash
# Sync to S3
make sync_data_to_s3

# Sync from S3
make sync_data_from_s3

# View DVC status
dvc status
dvc diff

# Push to remote storage
dvc push
dvc pull
```

### MLflow Commands
```bash
# Start MLflow UI (local)
mlflow ui --host 0.0.0.0 --port 5000

# View registered models
mlflow models list

# Load model for serving
mlflow models serve -m "models:/my_model/Staging" --port 5001
```

---

## 📈 Model Metrics & Performance

### Evaluation Results
```json
{
    "accuracy": 0.7778,      // 77.78% of predictions correct
    "precision": 0.7707,     // 77.07% of Happy predictions correct
    "recall": 0.7788,        // 77.88% of actual Happy instances found
    "auc": 0.8640            // 86.40% discrimination ability
}
```

### Interpretation:
- **Accuracy** is balanced - good overall performance
- **Precision ≈ Recall** - Model not biased toward either class
- **AUC = 0.8640** - Strong separation between classes

---

## 🔒 Security Notes

### Environment Variables
```bash
# Set DagsHub token (NEVER commit to Git)
export DAGSHUB_PAT=your_token

# Verify in .gitignore
echo "DAGSHUB_PAT" >> .gitignore
```

### Best Practices:
- ✅ Use environment variables for secrets
- ✅ Never commit API keys/tokens
- ✅ Use .gitignore for sensitive files
- ✅ Use DVC for data/model versioning
- ✅ Implement authentication for Flask endpoints

---

## 📝 Troubleshooting

### Common Issues

**Issue 1: DAGSHUB_PAT not set**
```bash
# Solution
export DAGSHUB_PAT=your_token
python src/model/model_evaluation.py
```

**Issue 2: vectorizer.pkl not found**
```bash
# Solution: Ensure feature_engineering ran first
python src/features/feature_engineering.py
```

**Issue 3: Model not found in registry**
```bash
# Solution: Register model first
python src/model/model_evaluation.py
python src/model/register_model.py
```

**Issue 4: Port 5000 already in use**
```bash
# Solution
python flask_app/app.py --port 5001
```

---

## 📚 Additional Resources

### Documentation Links:
- [DVC Documentation](https://dvc.org/doc)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [NLTK Documentation](https://www.nltk.org/)
- [Flask Documentation](https://flask.palletsprojects.com/)

### Related Projects:
- DagsHub Repository: https://dagshub.com/Abhishek9124/mlops-mini-project
- CampusX GitHub: https://github.com/campusx-official

---

## 👤 Author & License

**Author:** Abhishek  
**License:** MIT  
**Year:** 2025

### MIT License Summary:
- ✅ Can use commercially
- ✅ Can modify code
- ✅ Can distribute
- ❌ No warranty
- ❌ Limited liability

---

## 🎓 Learning Outcomes

By studying this project, you'll learn:

1. **MLOps Best Practices**
   - Pipeline orchestration (DVC)
   - Experiment tracking (MLflow)
   - Model versioning & registry

2. **Data Science Workflow**
   - Data ingestion & preprocessing
   - Feature engineering
   - Model training & evaluation

3. **NLP Fundamentals**
   - Text preprocessing techniques
   - Bag of Words vectorization
   - Sentiment classification

4. **Software Engineering**
   - Modular code design
   - Testing & CI/CD
   - Logging & error handling
   - Web application development

5. **DevOps Skills**
   - Environment configuration
   - Dependency management
   - Docker/containerization basics

---

**Last Updated:** December 16, 2025  
**Project Status:** Active Development  
**Version:** 1.0

---

Generated with detailed analysis of all project files.
