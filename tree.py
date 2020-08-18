from sklearn import tree

# [height, weight, shoe_size]
training_data = [[180, 80, 42], [169, 55, 36], [160, 60, 38]]
labels = ["male", "female", "female"]

# train the model using the training_data: learning simple decision rules
model = tree.DecisionTreeClassifier()
model.fit(training_data, labels)

# test data
test_data = [[185, 82, 44], [170, 75, 41], [160, 70, 38]]
predictions = model.predict(test_data)

results = zip(test_data, predictions)
for test, prediction in results:
    print(f"{test} {prediction}")
