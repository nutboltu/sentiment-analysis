import sklearn
from sklearn.datasets import load_svmlight_file
from sklearn.externals.joblib import Memory

mem = Memory("./mycache")

@mem.cache
def load_feature_file(filePath, num_of_features=None):
    data = load_svmlight_file(filePath, n_features=num_of_features)
    return data[0], data[1]
