import pandas as pd
import webbrowser
import os
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.externals import joblib


df = pd.read_csv("house_data_set.csv")
#########################################
## To see data set after using One Hot Encoding
#########################################
#html = df[0:100].to_html()
## Save the html to a temporary file
#with open("data.html", "w") as f:
#    f.write(html)
## Open the web page in our web browser
#full_filename = os.path.abspath("data.html")
#webbrowser.open("file://{}".format(full_filename))


del df['zip_code']
del df['unit_number']
del df['street_name']
del df['house_number']

features_df = pd.get_dummies(df, columns=['garage_type', 'city'])


#########################################
## To see data set after using One Hot Encoding
#########################################
#html = features_df[0:100].to_html()
## Save the html to a temporary file
#with open("data.html", "w") as f:
#    f.write(html)
## Open the web page in our web browser
#full_filename = os.path.abspath("data.html")
#webbrowser.open("file://{}".format(full_filename))


del features_df['sale_price']

# Creating Feature array
X = features_df.as_matrix()
y=df['sale_price'].as_matrix()


#Splitting Test and Training Data set
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=0)

#Fit to a Model & Set the HyperParameters
model = ensemble.GradientBoostingRegressor(
        n_estimators = 3000,
        learning_rate = 0.1,
        max_depth = 6,
        min_samples_leaf =3,
        max_features= 1,
        loss = 'huber'
        )

#########################################
##Without ParamGrid
#########################################

#Train Model after setting hyperparameter
model.fit(X_train, y_train)

#Save the Trained Model
joblib.dump(model,'trained_model.pkl')


#Training Error Rate
mse= mean_absolute_error(y_train, model.predict(X_train))
print(mse)

#Test Error Rate
mse=mean_absolute_error(y_test, model.predict(X_test))
print(mse)


##########################################
##With ParamGrid.
##########################################
##Setting ParamGrid
#param_grid={'n_estimators': [500,3000],
#        'learning_rate': [0.1],
#        'max_depth': [4,6],
#        'min_samples_leaf': [3],
#        'max_features': [1],
#        'loss': ['huber']
#                     
#     }
#
## Define the grid search we want to run. Run it with four cpus in parallel.
#gs_cv=GridSearchCV(model,param_grid,n_jobs=1)


## Run the grid search - on only the training data!
#gs_cv.fit(X_train,y_train)
#
#print(gs_cv.best_params_)
#
## Find the error rate on the training set using the best parameters
#mse = mean_absolute_error(y_train, gs_cv.predict(X_train))
#print("Training Set Mean Absolute Error: %.4f" % mse)
#
## Find the error rate on the test set using the best parameters
#mse = mean_absolute_error(y_test, gs_cv.predict(X_test))
#print("Test Set Mean Absolute Error: %.4f" % mse)
