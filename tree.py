import os
from sys import argv
import pandas as pd

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import mlflow
import mlflow.sklearn


if __name__ == "__main__":
    # parameters
    max_depth = int(argv[1]) if len(argv) > 1 else None
    criterion = argv[2] if len(argv) > 2 else "gini"

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.csv")
    data = pd.read_csv(data_path)
    print(data.head())

    # prepare the data
    # X contains the attributes
    X = data.drop(["Gender"], axis=1)
    # Y contains the labels
    y = data["Gender"]

    # Split data into training and test
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.20)

    # start and manage mlflow run
    with mlflow.start_run():

        # create decision tree model and train it
        model = tree.DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)
        model.fit(train_x, train_y)

        predict_y = model.predict(test_x)

        # evaluate
        accuracy = accuracy_score(test_y, predict_y)
        print(f"accuracy; {accuracy}")

        # https://en.wikipedia.org/wiki/Confusion_matrix
        cm = confusion_matrix(test_y, predict_y)
        print(f"confustion matrix: \n{cm}")
        tp = cm[0][0]
        tn = cm[1][1]
        fp = cm[0][1]
        fn = cm[1][0]

        # mlflow stuff
        mlflow.log_param("criterion", criterion)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("tp", tp)
        mlflow.log_metric("tn", tn)
        mlflow.log_metric("fp", fp)
        mlflow.log_metric("fn", fn)

        mlflow.sklearn.log_model(model, "model")
