# MLOps Quick Reference Guide

## DVC Commands Quick Reference

### Pipeline Execution
```bash
dvc repro                          # Run entire pipeline
dvc repro model_building          # Run specific stage
dvc repro --force                 # Force rerun all stages
dvc dag                           # Show pipeline structure
dvc status                        # Check what's changed
```

### Data Management
```bash
dvc add data/raw                  # Track data with DVC
dvc push                          # Upload to remote
dvc pull                          # Download from remote
dvc fetch                         # Download without checkout
dvc status -c                     # Check remote sync status
```

### Metrics & Experiments
```bash
dvc metrics show                  # Display metrics
dvc metrics diff                  # Compare metrics
dvc params show                   # Display parameters
dvc params diff                   # Compare parameters
dvc exp run                       # Run experiment
dvc exp show                      # Show all experiments
dvc exp compare exp1 exp2         # Compare experiments
```

### Caching & Cleanup
```bash
dvc cache dir                     # Show cache location
dvc cache dir --show-size         # Cache statistics
dvc cache verify                  # Verify cache integrity
dvc cache prune                   # Remove unused cache
dvc cache remove --not-in-remote  # Remove unsynced files
```

### Configuration
```bash
dvc remote list                   # Show all remotes
dvc remote list -v                # Show with details
dvc remote add -d myremote s3://bucket    # Add remote
dvc remote modify myremote auth basic     # Set auth
dvc config --list                 # Show all config
```

---

## GitHub Actions Commands

```bash
# List workflows
gh workflow list

# Run workflow manually
gh workflow run ml-pipeline.yml

# View runs
gh run list
gh run list --workflow=ci-pipeline.yml

# View specific run
gh run view RUN_ID

# View run logs
gh run view RUN_ID --log

# Cancel run
gh run cancel RUN_ID

# Watch run in real-time
gh run watch RUN_ID

# Rerun failed job
gh run rerun RUN_ID
```

---

## GitHub Secrets Management

```bash
# Add secret
gh secret set SECRET_NAME --body "secret_value"

# List secrets
gh secret list

# Delete secret
gh secret remove SECRET_NAME

# Required secrets:
# - AWS_ACCESS_KEY
# - AWS_SECRET_KEY
# - DAGSHUB_PAT
# - DOCKER_USERNAME (optional)
# - DOCKER_PASSWORD (optional)
```

---

## Local Development Workflow

```bash
# 1. Setup
python -m venv mlops-env
source mlops-env/Scripts/activate
pip install -r requirements.txt
pip install -e .

# 2. Initialize DVC
dvc init
dvc remote add -d myremote s3://bucket-name
dvc pull

# 3. Develop
# Edit files...
dvc repro                    # Test locally
dvc metrics show

# 4. Commit & Push
git add .
git commit -m "Your message"
git push                     # Triggers CI/CD

# 5. Monitor
gh run list
gh run view <RUN_ID> --log
```

---

## Model Training Pipeline

```bash
# Manual run (local)
python src/data/data_ingestion.py
python src/data/data_preprocessing.py
python src/features/feature_engineering.py
python src/model/model_building.py
python src/model/model_evaluation.py
python src/model/register_model.py

# Automated run (DVC)
dvc repro

# Scheduled run (GitHub Actions)
# Automatically runs on schedule (see .github/workflows/scheduled-training.yml)
```

---

## Flask Application

```bash
# Run locally
python flask_app/app.py

# Run with Gunicorn (production)
gunicorn -w 4 flask_app.app:app

# Test endpoint
curl -X POST http://localhost:5000/predict -d "text=I love this"

# Docker run
docker build -t mlops-app .
docker run -p 5000:5000 mlops-app
```

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_model.py -v
pytest tests/test_flask_app.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run linting
flake8 src
pylint src

# Security check
bandit -r src/
```

---

## Monitoring & Debugging

### Check Pipeline Status
```bash
dvc status                # What needs rerunning
dvc dag                   # Pipeline structure
dvc metrics show          # Current metrics
```

### View Logs
```bash
cat errors.log
cat transformation_errors.log
cat feature_engineering_errors.log
cat model_building_errors.log
cat model_evaluation_errors.log
cat model_registration_errors.log
```

### GitHub Actions Logs
```bash
gh run view <RUN_ID> --log      # View complete log
gh run view <RUN_ID> --log | grep "error"  # Find errors
```

---

## Environment Variables

```bash
# DagsHub/MLflow
export DAGSHUB_PAT=your_token
export MLFLOW_TRACKING_URI=https://dagshub.com/user/repo.mlflow
export MLFLOW_TRACKING_USERNAME=$DAGSHUB_PAT
export MLFLOW_TRACKING_PASSWORD=$DAGSHUB_PAT

# AWS (for S3)
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_REGION=us-east-1

# Python
export PYTHONUNBUFFERED=1
```

---

## File Structure

```
project/
├── .github/workflows/           # CI/CD automation
│   ├── ci-pipeline.yml         # Code quality & tests
│   ├── ml-pipeline.yml         # Model training
│   └── scheduled-training.yml  # Scheduled retraining
├── src/                        # Source code
│   ├── data/                  # Data scripts
│   ├── features/              # Feature engineering
│   ├── model/                 # Model training
│   └── visualization/         # Visualization
├── flask_app/                 # Web application
│   ├── app.py                # Flask app
│   ├── preprocessing_utility.py
│   └── templates/            # HTML templates
├── tests/                     # Unit tests
├── data/                      # Datasets
│   ├── raw/                  # Original data
│   ├── interim/              # Processed data
│   └── processed/            # Final features
├── models/                    # Trained models
├── reports/                   # Results & metrics
├── dvc.yaml                   # DVC pipeline
├── params.yaml                # Hyperparameters
├── requirements.txt           # Dependencies
├── setup.py                   # Package config
├── Dockerfile                 # Docker image
└── README.md                  # Documentation
```

---

## Troubleshooting Checklist

### Issue: Pipeline Won't Run
- [ ] Check DVC installation: `dvc version`
- [ ] Verify dvc.yaml syntax: `dvc dag`
- [ ] Check file paths exist
- [ ] Verify dependencies are installed

### Issue: DVC Remote Connection Fails
- [ ] Verify remote config: `dvc remote list -v`
- [ ] Check credentials: `dvc status -c`
- [ ] Test connectivity: `dvc push --dry`
- [ ] Check AWS/S3 permissions

### Issue: GitHub Actions Fails
- [ ] Check workflow syntax: `gh workflow list`
- [ ] Verify secrets are set: `gh secret list`
- [ ] View logs: `gh run view <ID> --log`
- [ ] Check Python version compatibility

### Issue: Model Not Found
- [ ] Verify model.pkl exists: `ls models/`
- [ ] Check vectorizer.pkl: `ls models/`
- [ ] Run feature engineering first: `python src/features/feature_engineering.py`

---

## Performance Optimization

### Speed Up Pipeline
```bash
# Skip caching for large files
dvc cache remove --not-in-remote

# Use faster remote (local directory for testing)
dvc remote add -d test-remote /tmp/dvc-storage

# Run specific stage only
dvc repro model_evaluation  # Only from this stage onward
```

### Reduce Storage Usage
```bash
# Check cache size
dvc cache dir --show-size

# Prune unused cache
dvc cache prune

# Remove old experiments
dvc exp gc --workspace
```

---

## Integration Examples

### With CI/CD
```yaml
# In GitHub Actions workflow
- name: Run pipeline
  run: |
    dvc repro
    dvc metrics show
    dvc push
```

### With Docker
```dockerfile
RUN pip install dvc
RUN dvc pull
RUN dvc repro
```

### With Kubernetes
```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: mlops-training
spec:
  schedule: "0 3 * * 0"
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: mlops
            image: mlops-app:latest
            command: ["dvc", "repro"]
```

---

## Resources

### Documentation
- [DVC Docs](https://dvc.org/doc)
- [GitHub Actions Docs](https://docs.github.com/en/actions)
- [MLflow Docs](https://mlflow.org/docs)
- [DagsHub Docs](https://dagshub.com/docs)

### Related Commands
```bash
# Check Git status
git status
git log --oneline

# View branch info
git branch -a
git remote -v

# Clean up
git gc
dvc cache prune
```

---

## Keyboard Shortcuts

### Terminal
```
Ctrl + C       Stop current command
Ctrl + Z       Suspend process
Ctrl + L       Clear screen
Ctrl + A       Move to line start
Ctrl + E       Move to line end
```

### Git
```
git add .                    Stage all changes
git commit -m "msg"          Commit with message
git push origin branch       Push to remote
git pull origin branch       Pull from remote
```

### DVC
```
dvc repro                    Run pipeline
dvc metrics show             View metrics
dvc push                     Upload artifacts
dvc pull                     Download artifacts
```

---

**Last Updated:** December 16, 2025
**Version:** 1.0
