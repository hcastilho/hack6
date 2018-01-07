import os
import graphviz

from exploration import modelling
from exploration.data_exploration import X_train, y_train, X_test, y_test
from exploration.modelling import cv
from exploration.utils import BASE_DIR
from sklearn import tree


def format_result(result):
    res = ('\n{result[name]}'
           '\nbest_params_: {result[rs].best_params_}'
           '\nscore: {result[score]}').format(result=result)
    return res


model = {
    'name': 'tree',
    'model': tree.DecisionTreeClassifier,
    'params': {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': range(1, 200),
        'min_samples_split': range(1, 200),
        # 'min_samples_leaf': range(1, 200),
        # 'min_weight_fraction_leaf': 0.0,
        # 'max_features': None,
        # 'random_state': None,
        # 'max_leaf_nodes': None,
        # 'min_impurity_decrease': 0.0,
        # 'min_impurity_split': None,
        # 'class_weight': None,
        # 'presort': False,
    },
}
result = modelling.run_hyper(model, X_train, y_train, X_test, y_test, cv)
print(format_result(result))


dot_data = tree.export_graphviz(
    result['estimator'],
    feature_names=X_test.columns,
    # class_names=['target'],
    out_file=None,
    filled=True,
    rounded=True,
    # special_characters=True,
)
graph = graphviz.Source(
    dot_data,
    directory=os.path.join(BASE_DIR, 'doc/report/img/'),
)
graph.render()
