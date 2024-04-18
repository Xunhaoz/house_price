import warnings

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


warnings.filterwarnings('ignore')

train_data = pd.read_csv('custom_train_data.csv', index_col='ID')
train_label = pd.read_csv('custom_train_label.csv', index_col='ID')
test_data = pd.read_csv('custom_test_data.csv', index_col='ID')

param_grid = {
    'n_estimators': [100, 200, 300, 500, 700],
    'max_depth': [20, 25, 30, 35, 40],
    'min_samples_split': [2, 3],
    'min_samples_leaf': [1, 2]
}

random_forest_regressor = RandomForestRegressor(criterion='absolute_error', n_jobs=-1)
grid_search = GridSearchCV(random_forest_regressor, param_grid=param_grid, cv=5, verbose=10)
grid_search.fit(train_data, train_label)

pd.DataFrame(grid_search.cv_results_).to_csv('cv_results_.csv')
print("Best parameters found: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)
