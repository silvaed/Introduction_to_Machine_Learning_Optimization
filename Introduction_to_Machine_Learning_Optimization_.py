
import pandas as pd

uri = "https://gist.githubusercontent.com/guilhermesilveira/e99a526b2e7ccc6c3b70f53db43a87d2/raw/1605fc74aa778066bf2e6695e24d53cf65f2f447/machine-learning-carros-simulacao.csv"
data = pd.read_csv(uri).drop(columns=["Unnamed: 0"], axis=1)
data.head()

# Terrible situation of "bad luck" where the classes are sorted by default

unlucky_data = data.sort_values("sold", ascending=True)
X_unlucky = unlucky_data[["price", "model_age","km_per_year"]]
y_unlucky = unlucky_data["sold"]
unlucky_data.head()

from sklearn.model_selection import cross_validate
from sklearn.dummy import DummyClassifier
import numpy as np

SEED = 301
np.random.seed(SEED)

model = DummyClassifier()
results = cross_validate(model, X_unlucky, y_unlucky, cv = 10, return_train_score=False)
mean = results['test_score'].mean()
std_dev = results['test_score'].std()
print("Accuracy with dummy stratified, 10 = [%.2f, %.2f]" % ((mean - 2 * std_dev)*100, (mean + 2 * std_dev) * 100))

from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier

SEED = 301
np.random.seed(SEED)

model = DecisionTreeClassifier(max_depth=2)
results = cross_validate(model, X_unlucky, y_unlucky, cv = 10, return_train_score=False)
mean = results['test_score'].mean()
std_dev = results['test_score'].std()
print("Accuracy with cross validation, 10 = [%.2f, %.2f]" % ((mean - 2 * std_dev)*100, (mean + 2 * std_dev) * 100))

# Generating random car model data for simulation of grouping when using our estimator

np.random.seed(SEED)
data['model'] = data.model_age + np.random.randint(-2, 3, size=10000)
data.model = data.model + abs(data.model.min()) + 1
data.head()

def print_results(results):
  mean = results['test_score'].mean() * 100
  deviation = results['test_score'].std() * 100
  print("Average Accuracy %.2f" % mean)
  print("Interval [%.2f, %.2f]" % (mean - 2 * deviation, mean + 2 * deviation))

# GroupKFold in a pipeline with StandardScaler and SVC

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold

SEED = 301
np.random.seed(SEED)

scaler = StandardScaler()
model = SVC()

pipeline = Pipeline([('transformation', scaler), ('estimator', model)])

cv = GroupKFold(n_splits = 10)
results = cross_validate(pipeline, X_unlucky, y_unlucky, cv = cv, groups = data.model, return_train_score=False)
print_results(results)

# GroupKFold to analyze how the model behaves with new groups

SEED = 301
np.random.seed(SEED)

cv = GroupKFold(n_splits = 10)
model = DecisionTreeClassifier(max_depth=2)
results = cross_validate(model, X_unlucky, y_unlucky, cv = cv, groups = data.model, return_train_score=False)
print_results(results)

from sklearn.tree import export_graphviz
import graphviz

model.fit(X_unlucky, y_unlucky)
features = X_unlucky.columns
dot_data = export_graphviz(model, out_file=None, filled=True, rounded=True,
                          class_names=["no","yes"],
                          feature_names = features)
graph = graphviz.Source(dot_data)
graph
