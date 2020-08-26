

Running a decision tree model with mlflow.

# development setup

```bash
# create virtual env
python -m venv .venv

# activate virtual env
source .venv/bin/activate

# install deps
pip install -r requirements.txt
```

# development run

```bash
$ python tree.py <max_depth> <criterion>
```


# mlflow run

```bash
# run without mlruns.db if you don't want to use the registry
# export MLFLOW_TRACKING_URI=sqlite:///mlruns.db
mlflow run .
```

# check results

```bash
# run without mlruns.db if you don't want to use the registry
mlflow ui --backend-store-uri sqlite:///mlruns.db
```
