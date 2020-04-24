import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt

pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.max_columns', 30)
pd.set_option('display.width', None)

appsData = pd.read_csv('./data/googleplaystore.csv')
print(appsData)
appsData.info()

""" EDA  PROCESS"""
appsData.dropna(inplace=True)
appsData.info()         # all data columns have equal number of values now


# Cleaning Category col into integers according to number of unique values
categoryVal = appsData["Category"].unique()
print('category values --> ', categoryVal)
# creating dictionary for key: value pair of category : corresponding number
category_dict = {categoryVal[i]: i for i in range(0, len(categoryVal))}
# creating new col in dataset for representing category type
appsData["Category_c"] = appsData["Category"].map(category_dict).astype(int)


# removing 'M' and 'k' from size col and converting str to float
def change_size(size):
    if 'M' in size:
        x = size[:-1]
        x = float(x)
        return(x)
    elif 'k' == size[-1:]:
        x = size[:-1]
        x = float(x) / 1000
        return(x)
    else:
        return None


# applying function to the dataset
appsData["Size"] = appsData["Size"].map(change_size)
# filling nan values in size column
appsData['Size'].fillna(method='ffill', inplace=True)


# changing '+' and ',' in Installs column
appsData['Installs'] = appsData['Installs'].apply(lambda x: str(x).replace('+', '') if '+' in str(x) else x)
appsData['Installs'] = appsData['Installs'].apply(lambda x: str(x).replace(',', '') if ',' in str(x) else x)
appsData['Installs'] = appsData['Installs'].apply(lambda x: float(x))


# adding new CATEGORICAL COLUMN for paid = 0 (false) in dataset
def type_cat(types):
    if types == 'Free':
        return 0
    else:
        return 1


appsData['Type'] = appsData['Type'].map(type_cat)


# cleaning of content rating classification
# unique values in content rating col
ratingL = appsData['Content Rating'].unique()
print(ratingL)
# creating dictionary for key: value pair of content rating : corresponding number
rating_dict = {ratingL[i]: i for i in range(0, len(ratingL))}
# creating new col in dataset for representing content rating type
appsData['Content Rating'] = appsData['Content Rating'].map(rating_dict).astype(int)

# drop unnecessary columns
appsData.drop(['App', 'Android Ver', 'Current Ver', 'Last Updated'], axis=1, inplace=True)

# cleaning the genres column
# unique values in content genres col
genresL = appsData['Genres'].unique()
print('\nthis is genresl -->', genresL)
# creating dictionary for key: value pair of genres : corresponding number
genres_dic = {genresL[i]: i for i in range(0, len(genresL))}
# creating new col in dataset for representing genres type
appsData['Genres_New'] = appsData['Genres'].map(genres_dic).astype(int)


# getting dummy values for price
def price_clean(price):
    if price == '0':
        return 0
    else:
        price = price[1:]
        price = float(price)
        return price


appsData['Price'] = appsData['Price'].map(price_clean).astype(int)

# converting reviews from str to int
appsData['Reviews'] = appsData['Reviews'].apply(lambda x: int(x))

# final data after EDA process
print(appsData)
appsData.info()

# In this instance, I created another dataframe that specifically created dummy
# values for each categorical instance in the dataframe, defined as df2

df2 = pd.get_dummies(data=appsData, columns=['Category'])
print(df2)

"""     EDS ENDS       """

"""     ML STARTS FROM HERE     """
# After our final checks for the preprocessing of our data, looks like we can start work!
# So the next question is what exactly are we doing and how are we doing it.
#
# So the goal of this instance is to see if we can use existing data provided(e.g. Size, no of reviews)
# to predict the ratings of the google applications.
# In other words, our dependent variable Y, would be the rating of the apps.
#
# One important factor to note is that the dependent variable Y, is a continuous
# variable(aka infinite no of combinations), as compared to a discrete variable.
# Naturally there are ways to convert our Y to a discrete variable but I decided to keep Y
# as a continuous variable for the purposes of this machine learning session.
#
# Next question, what models should we apply and how should we evaluate them?
#
# Model wise, I'm not too sure as well as there are like a ton of models out there that can be used for
# machine learning. Hence, I basically just chose the 3 most common models that I use, being
# linear regression, SVR, and random forest regressor.
#
# We technically run 4 regressions for each model used, as we consider one-hot vs interger encoded
# results for the category section, as well as including/excluding the genre section.
#
# We then evaluate the models by comparing the predicted results against the actual results graphically,
# as well as use the mean squared error, mean absolute error and mean squared log error as possible benchmarks.
#
# The use of the error term will be evaluated right at the end after running through all the models.

# let's use 3 different regression models with two different techniques on treating the categorical variable


# for evaluation of error term and
def evaluationmatrix(y_true, y_predict):
    print ('Mean Squared Error: '+ str(metrics.mean_squared_error(y_true,y_predict)))
    print ('Mean absolute Error: '+ str(metrics.mean_absolute_error(y_true,y_predict)))
    print ('Mean squared Log Error: '+ str(metrics.mean_squared_log_error(y_true,y_predict)))


# to add into results_index for evaluation of error term
def evaluationmatrix_dict(y_true, y_predict, name = 'Linear - Integer'):
    dict_matrix = {}
    dict_matrix['Series Name'] = name
    dict_matrix['Mean Squared Error'] = metrics.mean_squared_error(y_true,y_predict)
    dict_matrix['Mean Absolute Error'] = metrics.mean_absolute_error(y_true,y_predict)
    dict_matrix['Mean Squared Log Error'] = metrics.mean_squared_log_error(y_true,y_predict)
    return dict_matrix


#########################################################################################################
# linear regression model
# excluding genre label

from sklearn.linear_model import LinearRegression

X = appsData.drop(['Category', 'Rating', 'Genres', 'Genres_New'], axis=1)
print(X)
y = appsData.Rating
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
model = LinearRegression()
model.fit(X_train, y_train)
result = model.predict(X_test)

# Creation of results dataframe and addition of first entry
resultsdf = pd.DataFrame()
resultsdf = resultsdf.from_dict(evaluationmatrix_dict(y_test, result), orient='index')
resultsdf = resultsdf.transpose()

# dummy encoding
X_d = df2.drop(labels=['Rating', 'Genres', 'Category_c', 'Genres_New'], axis=1)
y_d = df2.Rating
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_d, y_d, test_size=0.30)
model_d = LinearRegression()
model_d.fit(X_train_d,y_train_d)
Results_d = model_d.predict(X_test_d)

# adding results into results dataframe
resultsdf = resultsdf.append(evaluationmatrix_dict(y_test_d,Results_d, name = 'Linear - Dummy'),ignore_index = True)


# plotting
plt.figure(figsize=(12, 7))
sns.regplot(x=result, y=y_test, color='teal', label='Integer', marker='x')
sns.regplot(x=Results_d, y=y_test, color='orange', label='dummy', marker='o')
plt.legend()
plt.title('Linear Model - Excluding Genres')
plt.xlabel('Predicted Rating')
plt.ylabel('Actual Rating')
plt.show()

print ('Actual mean of population:' + str(y.mean()))
print ('Integer encoding(mean) :' + str(result.mean()))
print ('Dummy encoding(mean) :'+ str(Results_d.mean()))
print ('Integer encoding(std) :' + str(result.std()))
print ('Dummy encoding(std) :'+ str(Results_d.std()))


# Including genre label

# Integer encoding
X = appsData.drop(labels = ['Category','Rating','Genres'],axis = 1)
y = appsData.Rating
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
model = LinearRegression()
model.fit(X_train,y_train)
Results = model.predict(X_test)

resultsdf = resultsdf.append(evaluationmatrix_dict(y_test,Results, name = 'Linear(inc Genre) - Integer'),ignore_index = True)

# dummy encoding

X_d = df2.drop(labels = ['Rating','Genres','Category_c'],axis = 1)
y_d = df2.Rating
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_d, y_d, test_size=0.30)
model_d = LinearRegression()
model_d.fit(X_train_d,y_train_d)
Results_d = model_d.predict(X_test_d)

resultsdf = resultsdf.append(evaluationmatrix_dict(y_test_d,Results_d, name = 'Linear(inc Genre) - Dummy'),ignore_index = True)
print('Including Genres')
print('Integer encoding(mean) :' + str(Results.mean()))
print('Dummy encoding(mean) :'+ str(Results_d.mean()))
print('Integer encoding(std) :' + str(Results.std()))
print('Dummy encoding(std) :'+ str(Results_d.std()))

# poltting
plt.figure(figsize=(12,7))
sns.regplot(Results,y_test,color='teal', label = 'Integer', marker = 'x')
sns.regplot(Results_d,y_test_d,color='orange',label = 'Dummy')
plt.legend()
plt.title('Linear model - Including Genres')
plt.xlabel('Predicted Ratings')
plt.ylabel('Actual Ratings')
plt.show()

#######################################################################################################
# SVM method
# Excluding genres
from sklearn import svm
# Integer encoding

X = appsData.drop(labels = ['Category','Rating','Genres','Genres_New'],axis = 1)
y = appsData.Rating
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

model2 = svm.SVR()
model2.fit(X_train, y_train)

Results2 = model2.predict(X_test)

resultsdf = resultsdf.append(evaluationmatrix_dict(y_test,Results2, name = 'SVM - Integer'),ignore_index = True)

# dummy based
X_d = df2.drop(labels=['Rating','Genres','Category_c','Genres_New'],axis = 1)
y_d = df2.Rating

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_d, y_d, test_size=0.30)

model2 = svm.SVR()
model2.fit(X_train_d, y_train_d)

Results2_d = model2.predict(X_test_d)

resultsdf = resultsdf.append(evaluationmatrix_dict(y_test_d,Results2_d, name = 'SVM - Dummy'),ignore_index = True)
print ('Integer encoding(mean) :' + str(Results2.mean()))
print ('Dummy encoding(mean) :'+ str(Results2_d.mean()))
print ('Integer encoding(std) :' + str(Results2.std()))
print ('Dummy encoding(std) :'+ str(Results2_d.std()))

# plotting
plt.figure(figsize=(12,7))
sns.regplot(Results2,y_test,color='teal', label = 'Integer', marker = 'x')
sns.regplot(Results2_d,y_test_d,color='orange',label = 'Dummy')
plt.legend()
plt.title('SVM model - excluding Genres')
plt.xlabel('Predicted Ratings')
plt.ylabel('Actual Ratings')
plt.show()


# Integer encoding, including Genres_c
model2a = svm.SVR()

X = appsData.drop(labels = ['Category','Rating','Genres'],axis = 1)
y = appsData.Rating

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

model2a.fit(X_train,y_train)

Results2a = model2a.predict(X_test)

# evaluation
resultsdf = resultsdf.append(evaluationmatrix_dict(y_test,Results2a, name = 'SVM(inc Genres) - Integer'),ignore_index = True)

# dummy encoding, including Genres_c
model2a = svm.SVR()

X_d = df2.drop(labels = ['Rating','Genres','Category_c'],axis = 1)
y_d = df2.Rating

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_d, y_d, test_size=0.30)

model2a.fit(X_train_d,y_train_d)

Results2a_d = model2a.predict(X_test_d)

# evaluation
resultsdf = resultsdf.append(evaluationmatrix_dict(y_test_d,Results2a_d, name = 'SVM(inc Genres) - Dummy'),ignore_index = True)
print(resultsdf)
print ('Integer encoding(mean) :' + str(Results2a.mean()))
print ('Dummy encoding(mean) :'+ str(Results2a_d.mean()))
print ('Integer encoding(std) :' + str(Results2a.std()))
print ('Dummy encoding(std) :'+ str(Results2a_d.std()))

# plotting
plt.figure(figsize=(12,7))
sns.regplot(Results2a,y_test,color='teal', label = 'Integer', marker = 'x')
sns.regplot(Results2a_d,y_test_d, color='orange', label = 'Dummy')
plt.legend()
plt.title('SVM model - including Genres')
plt.xlabel('Predicted Ratings')
plt.ylabel('Actual Ratings')
plt.show()

############################################################################################
# RANDOM FOREST REGRESSION
# excluding
from sklearn.ensemble import RandomForestRegressor

#Integer encodingqqq
X = appsData.drop(labels = ['Category', 'Rating', 'Genres', 'Genres_New'],axis = 1)
y = appsData.Rating
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
model3 = RandomForestRegressor()
model3.fit(X_train,y_train)
Results3 = model3.predict(X_test)

# evaluation
resultsdf = resultsdf.append(evaluationmatrix_dict(y_test,Results3, name = 'RFR - Integer'),ignore_index = True)

# dummy encoding
X_d = df2.drop(labels = ['Rating','Genres','Category_c','Genres_New'],axis = 1)
y_d = df2.Rating
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_d, y_d, test_size=0.30)
model3_d = RandomForestRegressor()
model3_d.fit(X_train_d,y_train_d)
Results3_d = model3_d.predict(X_test_d)

# evaluation
resultsdf = resultsdf.append(evaluationmatrix_dict(y_test,Results3_d, name = 'RFR - Dummy'),ignore_index = True)
print ('Integer encoding(mean) :' + str(Results3.mean()))
print ('Dummy encoding(mean) :'+ str(Results3_d.mean()))
print ('Integer encoding(std) :' + str(Results3.std()))
print ('Dummy encoding(std) :'+ str(Results3_d.std()))

# plotting
# plt.figure(figsize=(12,7))
# sns.regplot(Results3,y_test,color='teal', label = 'Integer', marker = 'x')
# sns.regplot(Results3_d,y_test_d,color='orange',label = 'Dummy')
# plt.legend()
# plt.title('RFR model - excluding Genres')
# plt.xlabel('Predicted Ratings')
# plt.ylabel('Actual Ratings')
# plt.show()

# RFR model is the best so far

######################################################################################################
# what influences the rating
# for integers
Feat_impt = {col: feat for col, feat in zip(X.columns,model3.feature_importances_)}
print(Feat_impt)

Feat_impt_df = pd.DataFrame.from_dict(Feat_impt, orient='index')
Feat_impt_df.sort_values(by=0, inplace=True)
Feat_impt_df.rename(index=str, columns={0:'Pct'}, inplace=True)
print(Feat_impt_df)

# plotting
plt.figure(figsize= (14,10))
Feat_impt_df.plot(kind = 'barh',figsize= (14,10),legend = False)
plt.show()


# for dummy
Feat_impt_d = {col: feat for col, feat in zip(X_d.columns,model3_d.feature_importances_)}
for col,feat in zip(X_d.columns,model3_d.feature_importances_):
    Feat_impt_d[col] = feat

Feat_impt_df_d = pd.DataFrame.from_dict(Feat_impt_d,orient = 'index')
Feat_impt_df_d.sort_values(by = 0, inplace=True)
Feat_impt_df_d.rename(index = str, columns={0:'Pct'},inplace = True)

plt.figure(figsize= (14,10))
Feat_impt_df_d.plot(kind = 'barh',figsize= (14,10),legend = False)
plt.tight_layout()
plt.show()


# Including Genres_new
#Integer encoding
X = appsData.drop(labels = ['Category','Rating','Genres'],axis = 1)
y = appsData.Rating
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
model3a = RandomForestRegressor()
model3a.fit(X_train,y_train)
Results3a = model3a.predict(X_test)

#evaluation
resultsdf = resultsdf.append(evaluationmatrix_dict(y_test,Results3a, name = 'RFR(inc Genres) - Integer'),ignore_index = True)

#dummy encoding

X_d = df2.drop(labels = ['Rating','Genres','Category_c'],axis = 1)
y_d = df2.Rating
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_d, y_d, test_size=0.30)
model3a_d = RandomForestRegressor()
model3a_d.fit(X_train_d,y_train_d)
Results3a_d = model3a_d.predict(X_test_d)

#evaluation
resultsdf = resultsdf.append(evaluationmatrix_dict(y_test,Results3a_d, name = 'RFR(inc Genres) - Dummy'),ignore_index = True)
print ('Integer encoding(mean) :' + str(Results3.mean()))
print ('Dummy encoding(mean) :'+ str(Results3_d.mean()))
print ('Integer encoding(std) :' + str(Results3.std()))
print ('Dummy encoding(std) :'+ str(Results3_d.std()))

# plotting
plt.figure(figsize=(12,7))
sns.regplot(Results3a,y_test,color='teal', label = 'Integer', marker = 'x')
sns.regplot(Results3a_d,y_test_d,color='orange',label = 'Dummy')
plt.legend()
plt.title('RFR model - including Genres')
plt.xlabel('Predicted Ratings')
plt.ylabel('Actual Ratings')
plt.show()


# for integer
Feat_impt = {}
for col,feat in zip(X.columns,model3a.feature_importances_):
    Feat_impt[col] = feat

Feat_impt_df = pd.DataFrame.from_dict(Feat_impt,orient = 'index')
Feat_impt_df.sort_values(by = 0, inplace = True)
Feat_impt_df.rename(index = str, columns = {0:'Pct'},inplace = True)

plt.figure(figsize= (14,10))
Feat_impt_df.plot(kind = 'barh',figsize= (14,10),legend = False)
plt.show()

#for dummy
Feat_impt_d = {}
for col,feat in zip(X_d.columns,model3a_d.feature_importances_):
    Feat_impt_d[col] = feat

Feat_impt_df_d = pd.DataFrame.from_dict(Feat_impt_d,orient = 'index')
Feat_impt_df_d.sort_values(by = 0, inplace = True)
Feat_impt_df_d.rename(index = str, columns = {0:'Pct'},inplace = True)

plt.figure(figsize= (14,10))
Feat_impt_df_d.plot(kind = 'barh',figsize= (14,10),legend = False)
plt.show()


# graphs for all types of models we implemented
resultsdf.set_index('Series Name', inplace = True)

plt.figure(figsize = (10,12))
plt.subplot(3, 1, 1)
resultsdf['Mean Squared Error'].sort_values(ascending = False).plot(kind = 'barh',color=(0.3, 0.4, 0.6, 1), title = 'Mean Squared Error')
plt.subplot(3, 1, 2)
resultsdf['Mean Absolute Error'].sort_values(ascending = False).plot(kind = 'barh',color=(0.5, 0.4, 0.6, 1), title = 'Mean Absolute Error')
plt.subplot(3, 1, 3)
resultsdf['Mean Squared Log Error'].sort_values(ascending = False).plot(kind = 'barh',color=(0.7, 0.4, 0.6, 1), title = 'Mean Squared Log Error')
plt.show()


"""             CONCLUSION          """
# Finally, looking at the results, it is not easy to conclude which model has the best predictive accuracy and lowest
# error term. Using this round of data as a basis, the dummy encoded SVM model including genres has the lowest overall
# error rates, followed by the integer encoded RFR model including genes. Yet, all models seem to be very close in terms
# of it's error term, so this result is likely to change.

# What is very surprising to me is how the RFR dummy model has such a significantly more error term compared to all the
# other models, even though on the surface it seemed to perform very similarly to the RFR integer model.

# Concluding thoughts It was pretty fun doing this project, using the three different machine learning models for
# continuous variables to see if it performed well in predictive analysis, based on the data that was provided.
# If you guys have any suggestions/comments please do feel free to post, as I'm still a beginner and want to learn more!
# Have a great and blessed day everyone!

