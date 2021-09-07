import  as np 
import pandas as pd 

# file paths to where the data is stored
train_data_path = './data/d_training_set.csv' 
test_data_path = './data/d_test_set.csv' 

train_data = pd.read_csv(train_data_path) 
test_data = pd.read_csv(test_data_path)




# a function to ease the preprocessing 
# steps, this will remove nan values 
# and returns the preprocessed data with 
# the lists of categorical and numerical 
# columns in the data and the row id 
# if the data to be preprocessed is the test set
def process_data(dframe, test=None):
    df = dframe
    print('Data Processing Started')
    if not test: 
        # drop the row_id column on the column axis
        df= df.drop('row_id', axis=1)
        cat_features = (df.dtypes == 'object') 
        cat_features = list(cat_features[cat_features].index) 
        num_features = (df.dtypes != 'object') 
        num_features = list(num_features[num_features].index) 

        # the slice of the data frame with categorical 
        # features
        df_cat = df[cat_features] 
        # the slice of the data frame with numerical features
        df_num  = df[num_features] 

        # fill empty categories row with 'Unknown'
        df_cat.fillna('Unknown', axis=1, inplace=True)
        # Additional numerical features in the dataset 
        num_features_2 = ['term', 'int_rate', 'emp_length', 'issue_d', 'revol_util'] 

        # Replacing the percentage symbol with an empty string 
        df_cat['int_rate'] = df_cat['int_rate'].replace({'%': ''}, regex=True) 
        # convert the string to a float 
        df_cat['int_rate'] = df_cat['int_rate'].astype(float) 
        # Replacing string symbols with integers 
        df_cat['term'] = df_cat['term'].replace({' 36 months': 36, ' 60 months': 60}, regex=True)
        df_cat['emp_length'] = df_cat['emp_length'].replace({'years': '', '10+': 10, 'year': '', '< 1 year': 1}, regex=True)
        # drop the 'issue_d' column as we deem it unnecessary for the model 
        df_cat.drop('issue_d', axis=1, inplace=True)
        df_cat['revol_util'] = df_cat['revol_util'].replace({'%': ''}, regex=True) 
        df_cat['revol_util'] = df_cat['revol_util'].replace({'Unknown': 0}, regex=True)
        df_cat['revol_util'] = df_cat['revol_util'].astype(float) 
        df_cat['term'] = df_cat['term'].astype(float)
        num_features_2.remove('issue_d')

        # numerical features extracted from the slice 
        # with categorical features
        df_num_2 = df_cat[num_features_2] 
        df_num = pd.concat([df_num, df_num_2], axis=1)
        # Filling missing variables with the mean of each column 
        df_num.fillna(df_num.mean(), inplace=True) 
        # drop the new numeric features from the categorical dataframe
        df_cat = df_cat.drop(num_features_2, axis=1)

        df_num['emp_length'] = df_num['emp_length'].replace({'< 1': 1}, regex=True) 
        df_num = df_num.replace({'Unknown': 0}, regex=True)

        df_combined = pd.concat([df_num, df_cat], axis=1)
        # categorical features (or attributes) 
        cat_attribs = list(df_cat.columns) 
        num_attribs = list(df_num.columns) 
        if 'repaid_loan' in num_attribs:
            num_attribs.remove('repaid_loan')
        print('*************Training data processed!!*************')
        return df_combined, cat_attribs, num_attribs

    elif test: 
        # if it's the test set 
        if 'row_id' in list(df.columns):
            
            row_id_col = df[['row_id']].copy() 
            df=df.drop('row_id', axis=1)
        cat_features = (df.dtypes == 'object') 
        cat_features = list(cat_features[cat_features].index) 
        num_features = (df.dtypes != 'object') 
        num_features = list(num_features[num_features].index) 

        df_cat = df[cat_features] 
        df_num  = df[num_features] 

        df_cat.fillna('Unknown', axis=1, inplace=True)
        # Additional numerical features in the dataset 
        num_features_2 = ['term', 'int_rate', 'emp_length', 'issue_d', 'revol_util'] 

        # Replacing the percentage symbol with an empty string 
        df_cat['int_rate'] = df_cat['int_rate'].replace({'%': ''}, regex=True) 
        # convert the string to a float 
        df_cat['int_rate'] = df_cat['int_rate'].astype(float) 
        # Replacing string symbols with integers 
        df_cat['term'] = df_cat['term'].replace({' 36 months': 36, ' 60 months': 60}, regex=True)
        df_cat['emp_length'] = df_cat['emp_length'].replace({'years': '', '10+': 10, 'year': '', '< 1 year': 1}, regex=True)
        # drop the 'issue_d' column as we deem it unnecessary for the model 
        df_cat.drop('issue_d', axis=1, inplace=True)
        df_cat['revol_util'] = df_cat['revol_util'].replace({'%': ''}, regex=True) 
        df_cat['revol_util'] = df_cat['revol_util'].replace({'Unknown': 0}, regex=True)
        df_cat['revol_util'] = df_cat['revol_util'].astype(float) 
        df_cat['term'] = df_cat['term'].astype(float)
        num_features_2.remove('issue_d')

        df_num_2 = df_cat[num_features_2] 
        df_num = pd.concat([df_num, df_num_2], axis=1)
        # Filling missing variables with the mean of each column 
        df_num.fillna(df_num.mean(), inplace=True) 
        # drop the new numeric features from the categorical dataframe
        df_cat = df_cat.drop(num_features_2, axis=1)

        df_num['emp_length'] = df_num['emp_length'].replace({'< 1': 1}, regex=True) 
        df_num = df_num.replace({'Unknown': 0}, regex=True)

        df_combined = pd.concat([df_num, df_cat], axis=1)
        # categorical features (or attributes) 
        cat_attribs = list(df_cat.columns) 
        num_attribs = list(df_num.columns) 
        print('*************Test data processed!!***************')
        return df_combined, cat_attribs, num_attribs,   row_id_col



# the processed version of the train_set
X_train_proc, cat_attribs, num_attribs = process_data(train_data)


# The slice of the dataset containing numeric features, plus the labels
X_train_num, y_train = X_train_proc[num_attribs], X_train_proc['repaid_loan']

# The processed version of the test_set
X_test_proc, _, _, row_id_col = process_data(test_data, test=True)
# The slice of the test set containing numeric features
X_test_num = X_test_proc[num_attribs]


from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler, OneHotEncoder 
from sklearn.compose import ColumnTransformer

num_pipeline = Pipeline([
    ('std_scaler', StandardScaler())
])

full_pipeline = ColumnTransformer([
    ('num' , num_pipeline, num_attribs), 

])

# The transformed version of both training and test dataset
train_data_prepared = full_pipeline.fit_transform(X_train_num)
test_data_prepared = full_pipeline.fit_transform(X_test_num)


# Trying to train a LinearRegression Model to see the results
from sklearn.linear_model import LinearRegression 

lin_reg = LinearRegression() 
lin_reg.fit(train_data_prepared, y_train)


# See how well the model performs on a sample 
# of the data
some_data = X_train_num.iloc[:5] 
some_labels = y_train.iloc[:5] 
some_data_prepared = full_pipeline.transform(some_data) 
print("Predictions: ", lin_reg.predict(some_data_prepared))


# mean_squared_error 
# to compute the error rate of the model
from sklearn.metrics import mean_squared_error 

loan_predictions = lin_reg.predict(train_data_prepared) 
lin_mse = mean_squared_error(y_train, loan_predictions) 
lin_rmse = np.sqrt(lin_mse) 
print('LR min_squared_error : {:.3f}'.format(lin_rmse))


from sklearn.metrics import mean_absolute_error 

lin_mae = mean_absolute_error(y_train, loan_predictions) 
print('LR mean squared error {}'.format(lin_mae))


# Training a DecisionTree
from sklearn.tree import DecisionTreeRegressor 

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(train_data_prepared, y_train)

# Make prediction on the training set
loan_predictions = tree_reg.predict(train_data_prepared) 
tree_mse = mean_squared_error(y_train, loan_predictions) 
tree_rmse = np.sqrt(tree_mse)
tree_mse


from sklearn.model_selection import cross_val_score 

scores = cross_val_score(tree_reg, train_data_prepared, y_train, scoring="neg_mean_squared_error", cv=10) 
tree_rmse_scores = np.sqrt(-scores)

def display_scores(scores):
    print("Scores: ", scores) 
    print("Mean: ", scores.mean()) 
    print("STD: ", scores.std()) 
    
display_scores(tree_rmse_scores)

# IMporting a random forest regressor
from sklearn.ensemble import RandomForestRegressor 

# Initilializing a RandomFOrestRegressor with 10 estimators 
# The random_state ensures that weights are initaliazed consistently
forest_reg = RandomForestRegressor(n_estimators=10, random_state=42) 
forest_reg.fit(train_data_prepared, y_train)


# Predictions of the forest regressor on the training set
forest_reg_predictions = forest_reg.predict(train_data_prepared) 
forest_mse = mean_squared_error(y_train, forest_reg_predictions) 
forest_rmse = np.sqrt(forest_mse) 
forest_rmse

# Cross validation score, 
# The model will be evaluated on 10 folds of 
# the shuffled training data
from sklearn.model_selection import cross_val_score 

forest_scores = cross_val_score(forest_reg, train_data_prepared, y_train, 
                               scoring="neg_mean_squared_error", cv=10) 
forest_rmse_scores = np.sqrt(-forest_scores) 
display_scores(forest_rmse_scores)


#Grid Search to find the best performing model
from sklearn.model_selection import GridSearchCV 

# Hyper parameter search space
param_grid = [
    # try 12 (3x4) combinations of hyperparameters 
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]}, 
    {'bootstrap' : [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}
]

forest_reg = RandomForestRegressor(random_state=42) 
# train accross 5 folds, that's a total of (12+6)*5=90 rounds of training 
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, 
                          scoring='neg_mean_squared_error', return_train_score=True) 
grid_search.fit(train_data_prepared, y_train)



print('[INFO]: \n')
print('********Best Parameters for GridSearch Cross Validation: {}'.format(grid_search.best_params_))

print('********Best Estimator for GridSearch Cross validation: {}'.format(grid_search.best_estimator_))




# cross validation results 
print('********[CROSS VALIDATION RESULTS]*********')
cvres = grid_search.cv_results_ 
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]): 
    print(np.sqrt(-mean_score), params)


from sklearn.model_selection import RandomizedSearchCV 
from scipy.stats import randint 

param_distribs = {
    'n_estimators': randint(low=1, high=200), 
    'max_features': randint(low=1, high=8),
}

forest_reg = RandomForestRegressor(random_state=42) 
rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs, 
                               n_iter=10, cv=5, scoring="neg_mean_squared_error", random_state=42) 
rnd_search.fit(train_data_prepared, y_train)

print('***********[RANDOMIZED SEARCH CROSS VALIDATION RESULTS]************')
cv_res = rnd_search.cv_results_ 
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]): 
    print(np.sqrt(-mean_score), params)


#feature_importances = grid_search.best_estimator_.feature_importances_


# Picking the best performing model to make predictions
final_tree_model = grid_search.best_estimator_ 
final_test_predictions = final_tree_model.predict(X_test_num)


print('[INFO]: ************ PREDICTONS ON THE TEST SET **************')
print('Final test_set predictions: {}'.format(final_test_predictions))


# Concatenate row_id and repaid_loan columns 
# save it to a csv file called loan_predictions.csv
predictions_pd = pd.DataFrame(final_test_predictions, columns=['repaid_loan'])
predictions_csv = pd.concat([row_id_col, predictions_pd], axis=1)
predictions_csv.to_csv('loan_predictions.csv', header=False, index=False)




# In the future if time is given we can explore neural networks 
# and other complex architecture to achieve the best results
from sklearn.neural_network import MLPRegressor 

mlp = MLPRegressor(solver='sgd', max_iter=100, activation='relu', 
                  random_state=42, learning_rate_init=0.01, 
                  batch_size=train_data_prepared.shape[0], momentum= 0.04)
print('[INFO] ****** Training a Multilayer Perceptron A.K.A Neural Network ******')
mlp.fit(train_data_prepared, y_train)



mlp_scores = cross_val_score(mlp, train_data_prepared, y_train, 
                               scoring="neg_mean_squared_error", cv=10) 
mlp_rmse_scores = np.sqrt(-forest_scores) 
print('[INFO] ********* MLP SCORE *********')
display_scores(mlp_rmse_scores)


# Lets try ensemble methods
from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score 
from sklearn.ensemble import AdaBoostRegressor 

num_trees = 30 
kfold = KFold(n_splits=10) 
model = AdaBoostRegressor(n_estimators=num_trees, random_state=42) 
results = cross_val_score(model, X_train_num, y_train, cv=kfold) 
print(results.mean())