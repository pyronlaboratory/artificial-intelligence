import tensorflow as tf
import tensorflow.contrib.learn as skflow
from sklearn import datasets, metrics

iris = datasets.load_iris()

classifier = skflow.LinearClassifier(feature_columns=[tf.contrib.layers.real_valued_column("", dimension=iris.data.shape[1])], n_classes=3)
classifier.fit(iris.data, iris.target)
score = metrics.accuracy_score(iris.target,classifier.predict(iris.data))

print("Accuracy: %f" %score)
