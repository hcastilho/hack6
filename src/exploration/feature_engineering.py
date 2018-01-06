import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets, svm
from sklearn.feature_selection import SelectPercentile, f_classif

# http://scikit-learn.org/stable/modules/feature_selection.html

# #############################################################################
# Univariate feature selection with F-test for feature scoring
# We use the default selection function: the 10% most significant features

# NOTE: F-test assumes normal distribution of data and cannot be used in
# binary data
# https://en.wikipedia.org/wiki/F-test

# calculating correlations between data and target only tells me that there
#  is no linear relationship, not that there is no information given by the
#  data
# I guess the problem would be similar using ANOVA f-tests


#
# hugo.lopes 3:11 PM
# @hcastilho you can use several feature selection algorithms that helps you determine if there is _something there_ but use them with caution. You can use chi-squared test for categorical variables
# 3:11
# kruskal-wallis for numericals
# 3:11
# and check the p-value of these tests
# 3:12
# however, these tests generally fail when your data is not linear
# 3:12
# for non-linearity you can use mutual information, which is related with entropy
# 3:12
# http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html   -> for classification problems
# 3:13
# these type of algorithms, I would use when we have a lot of features and we want to reduce our dimensionality space
# 3:15
# you could also use Random forests or Gradient Boosting and remove iteratively the less important feature. However, there are also limitations on this strategy. For instance, you should really understand the meaning of the "feature importance" in those algorithms and be careful with correlated features

X = dataset.drop(['target'], axis=1)
y = dataset['target']


selector = SelectPercentile(f_classif, percentile=10)
selector.fit(X, y)

# Remove zeros
zero_idx = [idx for idx, value in enumerate(selector.pvalues_) if value == 0.0]
scores = [value for value in selector.pvalues_ if value != 0.0]
print(X.columns[zero_idx])
X = X.drop(X.columns[zero_idx], axis=1)

res = sorted(zip(scores, X.columns), key=lambda x: x[0])

X_indices = np.arange(X.shape[-1])
plt.bar(X_indices - .45, scores, width=.2,
        label=r'Univariate score ($-Log(p_{value})$)', color='darkorange',
        edgecolor='black')
plt.show()

