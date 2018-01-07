from exploration import modelling
from exploration.data_exploration import X_train, y_train, X_test, y_test
from exploration.modelling import cv
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
        # 'criterion': 'gini',
        # 'splitter': 'best',
        # 'max_depth': None,
        # 'min_samples_split': 2,
        # 'min_samples_leaf': 1,
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
results.append(result)
print(format_result(result))
