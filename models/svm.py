## SVM MODEL

# import sklearn
# from sklearn.model_selection import StratifiedKFold
# from sklearn import svm
# skf = StratifiedKFold(n_splits=4)

# parameters = [1 ,2 , 3]
  
# for c in parameters:
#   best_acc = - 1
#   model = svm.SVC(kernel='linear', C=c, cache_size=10000)
#   for train, test in skf.split(data, target):
#     classifier = model.fit(data[train], target[train])
#     print("%s %s" % (train, test))
#     accuracy = classifier.score(data[test], target[test])
#     # accuracy = sklearn.metrics.accuracy_score(target[test], y_pred)
#     if (accuracy > best_acc):
#      best_acc = accuracy
#     print(accuracy)
# print("n = %s  acc = %s" %( c,  best_acc))
