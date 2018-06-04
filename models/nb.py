import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB

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
  best_alpha = -1
  for alpha in range(1, 10):
    model = MultinomialNB(alpha = alpha)
    accuracy, classifier = cross_validation(model, data, target)
    print("alpha = %s  accuracy = %s" %( alpha,  accuracy))
    if (accuracy > best_accuracy):
      best_accuracy = accuracy
      best_classifier = classifier
      best_alpha = alpha
  print("Best classifier found for alpha = %s with accuracy = %s" %(best_alpha, best_accuracy))    
  return best_classifier    
