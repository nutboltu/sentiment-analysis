from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np

from models import nb, knn
from loader import load_feature_file
from writer import write_results

#Reading data
print("Reading Training Data...")
data, target = load_feature_file("./data/train/labeledBow.feat")

#Pre processing
print("Preprocessing Data...")
tfidf_transformer = TfidfTransformer()
train_data = tfidf_transformer.fit_transform(data)
for x in np.nditer(target, op_flags=['readwrite']):
  x[...] = 1 if x > 5 else 0

# Model Training
## Naive Bayes
print("Training Naive Bayes Model...")
nbClassifier = nb.find_model(train_data, target)
## KNN 
print("Training K-nearest Neighbor Model ...")
knnClassifier = knn.find_model(data,target)

# Prediction and Display of Results
data, target = load_feature_file("./data/test/labeledBow.feat", num_of_features=train_data.shape[1])
test_data = tfidf_transformer.transform(data)

## Predicting with Naive Bayes
print('Predicting with Naive Bayes...')
results = nbClassifier.predict(test_data)
write_results('nb.csv', results)

## Predicting with KNN
print('Predicting with KNN...')
results = knnClassifier.predict(test_data)
write_results('knn.csv', results)

print('Prediction Completed!')