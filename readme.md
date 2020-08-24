

Running a decision tree model with mlflow.

# setup

```bash
# create virtual env
python -m venv .venv

# activate virtual env
source .venv/bin/activate

# install deps
pip install -r requirements.txt
```

# run

```bash
$ python tree.py <max_depth> <criterion>
```

# check results

```bash
mlflow ui
```