import tensorflow as tf
from sklearn import datasets, metrics

iris = datasets.load_iris()
classifier = skflow.TensorFlowLinearClassifier(hidden_units=[10,20,10],n_classes=3)

classifier.fit(iris.data, iris_target)
score = metrics.accuracy_score(iris_target,classifier.predict(iris_data))

print("Accuracy: %f" %score)
