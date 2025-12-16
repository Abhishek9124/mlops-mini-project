# MLOps Foundation Setup Guide

## Table of Contents
1. [DVC Setup](#dvc-setup)
2. [GitHub Actions Setup](#github-actions-setup)
3. [DagsHub Integration](#dagshub-integration)
4. [Complete Workflow](#complete-workflow)

---

## DVC Setup

### Prerequisites
- Git repository initialized
- Python 3.10+ installed
- AWS account (for S3 remote) or alternative storage

### Step 1: Install DVC

```bash
# Using pip
pip install dvc

# Using conda
conda install -c conda-forge dvc

# Verify installation
dvc version
```

### Step 2: Initialize DVC in Repository

```bash
# Must be in Git repository
cd c:\Users\ADMIN\OneDrive\Desktop\CampusX\mlops-mini-project

# Initialize DVC
dvc init

# Verify initialization
git status
# Should show: .dvc/ directory added
```

### Step 3: Configure Remote Storage (S3)

#### Option A: AWS S3 (Recommended)

```bash
# Add S3 remote
dvc remote add -d myremote s3://your-bucket-name/mlops-data

# Configure AWS credentials
dvc remote modify myremote --local access_key_id YOUR_ACCESS_KEY
dvc remote modify myremote --local secret_access_key YOUR_SECRET_KEY

# Optional: Set AWS region
dvc remote modify myremote region us-east-1

# Verify remote configuration
dvc remote list -v
```

#### Option B: DagsHub (MLOps Platform)

```bash
# Create account at https://dagshub.com
# Create new repository

# Add DagsHub as remote
dvc remote add -d myremote 'https://dagshub.com/YOUR_USERNAME/mlops-mini-project.dvc'

# Set authentication
dvc remote modify myremote auth basic
dvc remote modify myremote --local username YOUR_USERNAME
dvc remote modify myremote --local password YOUR_DAGSHUB_TOKEN

# Test connection
dvc status
```

### Step 4: Add Data to DVC

```bash
# Add data directory to DVC
dvc add data/raw

# Commit DVC files to Git
git add data/raw.dvc .gitignore
git commit -m "Add DVC tracking for raw data"

# Push data to remote
dvc push
```

### Step 5: Verify DVC Pipeline

```bash
# View pipeline structure
dvc dag

# Run entire pipeline
dvc repro

# Check status
dvc status

# View metrics
dvc metrics show
```

---

## GitHub Actions Setup

### Step 1: Create Secrets

Go to GitHub Repository → Settings → Secrets and variables → Actions

Add the following secrets:

```
AWS_ACCESS_KEY          (AWS Access Key ID)
AWS_SECRET_KEY          (AWS Secret Access Key)
DAGSHUB_PAT            (DagsHub Personal Access Token)
DAGSHUB_USERNAME       (DagsHub username)
```

**Command to add secrets using GitHub CLI:**
```bash
# Install GitHub CLI
# https://cli.github.com/

# Authenticate
gh auth login

# Add secrets
gh secret set AWS_ACCESS_KEY --body "YOUR_ACCESS_KEY"
gh secret set AWS_SECRET_KEY --body "YOUR_SECRET_KEY"
gh secret set DAGSHUB_PAT --body "YOUR_DAGSHUB_TOKEN"
```

### Step 2: Create Workflow Files

The following files have been created in `.github/workflows/`:

1. **ci-pipeline.yml** - Code quality and testing
2. **ml-pipeline.yml** - ML training and evaluation
3. **scheduled-training.yml** - Automated retraining

### Step 3: Verify Workflows

```bash
# List all workflows
gh workflow list

# View recent runs
gh run list

# Check specific workflow
gh run list --workflow=ci-pipeline.yml
```

---

## DagsHub Integration

### Step 1: Create DagsHub Account

1. Go to https://dagshub.com
2. Sign up with GitHub (recommended)
3. Create new repository "mlops-mini-project"

### Step 2: Configure MLflow Tracking

```bash
# Set environment variable
export DAGSHUB_PAT=your_dagshub_token

# Verify in Python
import os
os.getenv('DAGSHUB_PAT')
```

### Step 3: Configure DVC Remote on DagsHub

```bash
# Add DagsHub as DVC remote
dvc remote add -d dagshub 'https://dagshub.com/YOUR_USERNAME/mlops-mini-project.dvc'

dvc remote modify dagshub auth basic
dvc remote modify dagshub --local username YOUR_USERNAME
dvc remote modify dagshub --local password YOUR_DAGSHUB_TOKEN

# Set as default
dvc remote default dagshub
```

### Step 4: Configure MLflow on DagsHub

In your Python code:

```python
import mlflow
import os

# Set MLflow tracking URI
mlflow.set_tracking_uri('https://dagshub.com/YOUR_USERNAME/mlops-mini-project.mlflow')

# Set credentials
os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv('DAGSHUB_PAT')
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('DAGSHUB_PAT')
```

### Step 5: Monitor Experiments

1. Visit: `https://dagshub.com/YOUR_USERNAME/mlops-mini-project`
2. Click on "Experiments" tab
3. View all MLflow experiments and metrics

---

## Complete Workflow

### Workflow Diagram

```
┌─────────────────────────────────┐
│  Developer Pushes Code to Git   │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  GitHub Actions Triggered       │
│  - ci-pipeline.yml (Code tests) │
└──────────────┬──────────────────┘
               │
               ├─→ Lint code (flake8)
               ├─→ Format check (black)
               ├─→ Security scan (bandit)
               ├─→ Run unit tests (pytest)
               └─→ Upload coverage
               │
               ▼
┌─────────────────────────────────┐
│  ml-pipeline.yml Triggered      │
│  - DVC repro                    │
│  - Train model                  │
│  - Evaluate metrics             │
└──────────────┬──────────────────┘
               │
               ├─→ Pull data (dvc pull)
               ├─→ Run pipeline (dvc repro)
               ├─→ Push artifacts (dvc push)
               ├─→ Register model (MLflow)
               └─→ Commit metrics
               │
               ▼
┌─────────────────────────────────┐
│  Scheduled Training             │
│  - Daily/Weekly retraining      │
│  - Metrics validation           │
│  - Auto-deployment              │
└─────────────────────────────────┘
```

### Step-by-Step Execution

#### 1. Local Development

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/mlops-mini-project
cd mlops-mini-project

# Create virtual environment
python -m venv mlops-env
source mlops-env/Scripts/activate  # Windows

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Initialize DVC
dvc init

# Configure DVC remote
dvc remote add -d myremote s3://your-bucket/path
# OR
dvc remote add -d myremote https://dagshub.com/YOUR_USERNAME/mlops-mini-project.dvc

# Pull data
dvc pull

# Run pipeline locally
dvc repro
```

#### 2. Make Changes

```bash
# Edit source files or parameters
# Example: Change params.yaml
vim params.yaml
# Increase max_features from 1000 to 2000

# Test locally
dvc repro
dvc metrics show

# View metrics
cat reports/metrics.json
```

#### 3. Commit and Push

```bash
# Stage changes
git add src/ params.yaml dvc.lock

# Commit
git commit -m "Increase vocabulary size for better performance"

# Push to GitHub
git push origin main
```

#### 4. GitHub Actions Runs Automatically

```
On push → ci-pipeline.yml runs (code quality + tests)
If main branch & params.yaml changed → ml-pipeline.yml runs (train model)
```

#### 5. Review Results

```bash
# Visit GitHub Actions tab
gh run list

# View specific run
gh run view RUN_ID --log

# View metrics report in GitHub
# (Posted as PR comment automatically)
```

#### 6. Scheduled Retraining

```
Daily at 2 AM UTC → scheduled-training.yml runs
├─→ Pull latest data
├─→ Retrain model
├─→ Validate metrics
└─→ Push updated model

Weekly at 3 AM Sunday → Full comprehensive retraining
```

---

## Quick Start Commands

### All-in-One Setup

```bash
# 1. Initialize DVC
dvc init

# 2. Configure remote
dvc remote add -d myremote s3://bucket-name/path

# 3. Create workflows directory
mkdir -p .github/workflows

# 4. Copy workflow files (already created)
# Files are in .github/workflows/

# 5. Add secrets in GitHub UI
# Settings → Secrets → Add AWS_ACCESS_KEY, AWS_SECRET_KEY, DAGSHUB_PAT

# 6. Push to GitHub
git add .github/ Dockerfile
git commit -m "Add CI/CD pipelines"
git push origin main

# 7. Monitor
gh run list
```

---

## Common Issues & Solutions

### Issue 1: "DAGSHUB_PAT not set"

```bash
# Solution
export DAGSHUB_PAT=your_token
python src/model/model_evaluation.py
```

### Issue 2: "Failed to push to remote"

```bash
# Check remote configuration
dvc remote list -v

# Verify credentials
dvc status

# Test push
dvc push --dry  # Dry run first

dvc push  # Actually push
```

### Issue 3: "GitHub Actions fails with DVC error"

In workflow file, add:

```yaml
- name: Configure DVC
  run: |
    dvc remote add -d myremote s3://bucket
    dvc remote modify myremote --local access_key_id ${{ secrets.AWS_ACCESS_KEY }}
    dvc remote modify myremote --local secret_access_key ${{ secrets.AWS_SECRET_KEY }}
```

### Issue 4: "Port 5000 already in use"

```bash
# Find process using port 5000
lsof -i :5000

# Kill process
kill -9 <PID>

# Or use different port
python flask_app/app.py --port 5001
```

---

## Monitoring & Maintenance

### Weekly Checklist

- [ ] Check GitHub Actions runs (gh run list)
- [ ] Review model metrics (dvc metrics show)
- [ ] Monitor DVC storage usage (dvc cache dir --show-size)
- [ ] Check for failed workflow runs
- [ ] Review code coverage reports

### Monthly Tasks

```bash
# Clean up old cache
dvc cache prune

# Review experiment history
dvc exp show

# Update dependencies
pip list --outdated
pip install --upgrade pip setuptools wheel

# Check DVC version
dvc version
```

### Useful Monitoring Commands

```bash
# View all metrics
dvc metrics show

# Compare metrics across commits
dvc metrics diff main

# View cache statistics
dvc cache dir --show-size

# Verify remote
dvc status -c

# List all experiments
dvc exp show
```

---

## Summary

### Foundation Components

| Component | Purpose | Setup | Monitoring |
|-----------|---------|-------|-----------|
| **DVC** | Data versioning | `dvc init` | `dvc status` |
| **Git** | Code versioning | `git init` | `git log` |
| **GitHub Actions** | CI/CD automation | Upload `.github/workflows/` | `gh run list` |
| **DagsHub** | MLflow tracking | Set `DAGSHUB_PAT` | Website UI |
| **S3** | Artifact storage | Configure remote | `dvc push/pull` |

### Commands You'll Use Daily

```bash
# Local development
dvc repro                    # Train model
dvc metrics show            # View results
git add .
git commit -m "Message"
git push                    # Triggers CI/CD

# Monitoring
gh run list                 # View runs
dvc exp show               # View experiments
dvc status                 # Check pipeline status

# Maintenance
dvc cache prune            # Clean storage
dvc push                   # Backup data
dvc pull                   # Get latest data
```

---

**Setup Complete!** Your MLOps foundation is now ready. 🎉

Next steps:
1. Run `git push` to trigger CI/CD
2. Monitor `gh run list` for workflow execution
3. Check metrics in GitHub Actions logs
4. Visit DagsHub dashboard for experiment tracking
