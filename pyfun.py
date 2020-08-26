from mlflow import pyfunc
import pandas as pd

run_id = "ab8008b2b2cb433a8508a9a60ef638df"
pyfunc_uri = f"runs:/{run_id}/model"
pyfunc_model = pyfunc.load_model(pyfunc_uri)
print(f"Loading the scikit-learn model ({pyfunc_uri}) = as PyFunc Model")

# predict
df = pd.DataFrame([[189, 87, 2]])
predict = pyfunc_model.predict(df)
print(predict)
