import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

skf = StratifiedKFold(n_splits=10)

parameters = [2, 3 , 5]
number_of_folds = 10

def cross_validation(model, data, target):
  skf = StratifiedKFold(n_splits=number_of_folds)
  best_accuracy = - 1
  best_classifier = None
  for train, test in skf.split(data, target):
      classifier = model.fit(data[train], target[train])
      y_predicted = classifier.predict(data[test])
      accuracy = sklearn.metrics.accuracy_score(target[test], y_predicted)
      if (accuracy > best_accuracy):
        best_accuracy = accuracy
        best_classifier = classifier
  return best_accuracy, best_classifier

def find_model(data, target):
  best_accuracy = -1
  best_classifier = None
  best_neighbor = -1
  for n in parameters:
    model = KNeighborsClassifier(n_neighbors=n)
    accuracy, classifier = cross_validation(model, data, target)
    print("n = %s  accuracy = %s" %( n,  accuracy))
    if (accuracy > best_accuracy):
      best_accuracy = accuracy
      best_classifier = classifier
      best_neighbor = n
  print("Best classifier found for neighbor = %s with accuracy = %s" %(best_neighbor, best_accuracy))    
  return best_classifier 

