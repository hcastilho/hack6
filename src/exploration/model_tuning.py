# model = {
#     'name': 'random_forest',
#     'model': ensemble.RandomForestClassifier,
#     'params': {
#         'n_estimators': range(400),  # 10
#         'criterion': ['gini', 'entropy'],
#         # 'max_depth': None,
#         'min_samples_split': range(400),  # 2
#         # 'min_samples_leaf': 1,
#         # 'min_weight_fraction_leaf': 0.0,
#         # 'max_features': 'auto',
#         # 'max_leaf_nodes': None,
#         # 'min_impurity_decrease': 0.0,
#         # 'min_impurity_split': None, # Deprecated
#         # 'bootstrap': True,
#         # 'oob_score': False,
#         # 'random_state': None,
#         # 'verbose': 0,
#         # 'warm_start': False,
#         # 'class_weight': None
#
#     },
# }
# result = modelling.run_hyper(model, X_train, y_train, X_test, y_test, cv)
# print(format_result(result))
