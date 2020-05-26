#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


train = pd.read_csv("train.csv")
holdout = pd.read_csv("test.csv")


# In[3]:


print(holdout.head())


# In[4]:


# %load functions.py
def process_missing(df):
    """Handle various missing values from the data set

    Usage
    ------

    holdout = process_missing(holdout)
    """
    df["Fare"] = df["Fare"].fillna(train["Fare"].mean())
    df["Embarked"] = df["Embarked"].fillna("S")
    return df

def process_age(df):
    """Process the Age column into pre-defined 'bins' 

    Usage
    ------

    train = process_age(train)
    """
    df["Age"] = df["Age"].fillna(-0.5)
    cut_points = [-1,0,5,12,18,35,60,100]
    label_names = ["Missing","Infant","Child","Teenager","Young Adult","Adult","Senior"]
    df["Age_categories"] = pd.cut(df["Age"],cut_points,labels=label_names)
    return df

def process_fare(df):
    """Process the Fare column into pre-defined 'bins' 

    Usage
    ------

    train = process_fare(train)
    """
    cut_points = [-1,12,50,100,1000]
    label_names = ["0-12","12-50","50-100","100+"]
    df["Fare_categories"] = pd.cut(df["Fare"],cut_points,labels=label_names)
    return df

def process_cabin(df):
    """Process the Cabin column into pre-defined 'bins' 

    Usage
    ------

    train process_cabin(train)
    """
    df["Cabin_type"] = df["Cabin"].str[0]
    df["Cabin_type"] = df["Cabin_type"].fillna("Unknown")
    df = df.drop('Cabin',axis=1)
    return df

def process_titles(df):
    """Extract and categorize the title from the name column 

    Usage
    ------

    train = process_titles(train)
    """
    titles = {
        "Mr" :         "Mr",
        "Mme":         "Mrs",
        "Ms":          "Mrs",
        "Mrs" :        "Mrs",
        "Master" :     "Master",
        "Mlle":        "Miss",
        "Miss" :       "Miss",
        "Capt":        "Officer",
        "Col":         "Officer",
        "Major":       "Officer",
        "Dr":          "Officer",
        "Rev":         "Officer",
        "Jonkheer":    "Royalty",
        "Don":         "Royalty",
        "Sir" :        "Royalty",
        "Countess":    "Royalty",
        "Dona":        "Royalty",
        "Lady" :       "Royalty"
    }
    extracted_titles = df["Name"].str.extract(' ([A-Za-z]+)\.',expand=False)
    df["Title"] = extracted_titles.map(titles)
    return df

def create_dummies(df,column_name):
    """Create Dummy Columns (One Hot Encoding) from a single Column

    Usage
    ------

    train = create_dummies(train,"Age")
    """
    dummies = pd.get_dummies(df[column_name],prefix=column_name)
    df = pd.concat([df,dummies],axis=1)
    return df


def apply_all(df):
    df_1 = process_missing(df)
    df_1 = process_age(df_1)
    df_1 = process_fare(df_1)
    df_1 = process_titles(df_1)
    df_1 = process_cabin(df_1)
    
    df_1 = create_dummies(df_1, ["Age_categories", "Fare_categories", "Title", "Cabin_type", "Sex"])
    return df_1 


# In[5]:


train = apply_all(train)
holdout = apply_all(holdout)


# # Build Kaggle Workflow
The Kaggle workflow we are going to build will combine all of these into a process.
Data exploration, to find patterns in the data
Feature engineering, to create new features from those patterns or through pure experimentation
Feature selection, to select the best subset of our current set of features
Model selection/tuning, training a number of models with different hyperparameters to find the best performer.
# In[6]:


explore_col = ["SibSp", "Parch"]
train[explore_col].info()


# In[7]:


#Using histograms to view the distribution of values in the columns
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


explore = train[explore_col].copy()
explore.plot.hist(alpha=0.5, bins=10, legend=True)

plt.show()


# In[8]:


explore["familysize"] = explore["SibSp"] + explore["Parch"]
explore.plot.hist(alpha=0.5, bins=10)
plt.xticks(range(11))
plt.show()


# Use pivot tables to look at the survival rate for different values of the columns
# The SibSp column shows the number of siblings and/or spouses each passenger had on board, while the Parch columns shows the number of parents or children each passenger had onboard. Neither column has any missing values.
# 
# The distribution of values in both columns is skewed right, with the majority of values being zero.

# In[9]:


import numpy as np

explore["Survived"] = train["Survived"]

for col in explore.columns.drop("Survived"):
    pivot = explore.pivot_table(index=col, values="Survived")
    #by default, the aggrefunction is numpy.mean()
    pivot.plot.bar(ylim=(0,1), yticks=np.arange(0,1,.1))
    plt.axhspan(.3,.6, alpha=0.2, color='red')
    plt.show()


# Looking at the survival rates of the the combined family members, you can see that few of the over 500 passengers with no family members survived, while greater numbers of passengers with family members survived.

# # Engineering New Features

# Based of this, we can come up with an idea for a new feature - was the passenger alone. This will be a binary column containing the value:
# 
# 1 if the passenger has zero family members onboard
# 0 if the passenger has one or more family members onboard

# In[10]:


def isalone(df):
    df["familysize"] = df[["SibSp","Parch"]].sum(axis=1)
    df["isalone"] = 0
    #initiate a column wiht values as 0
    df.loc[df["familysize"]==0, "isalone"] = 1
    df = df.drop("familysize", axis=1)    
    return df


# In[11]:


train = isalone(train)
holdout = isalone(holdout)


# # Feature Selection Using Sklearn

# In[13]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV


def select_features(df):
    #Removes any non-numeric columns or columns containing null values
    df = df.select_dtypes([np.number]).dropna(axis=1)
    all_X = df.drop(["PassengerId","Survived"], axis=1)
    all_y = df["Survived"]
    
    clf = RandomForestClassifier(random_state=1)
    selector = RFECV(clf, cv=10)
    #split in 10 fold cross validation
    selector.fit(all_X, all_y)
    
    best_columns = list(all_X.columns[selector.support_])
    print("Best Columns are \n"+"-"*12+"\n{0}\n".format(best_columns))    
    return best_columns
          
cols = select_features(train)          


# # Seleting and Tuning Different Algorithms
# #Grid_Search To Find Best Model

# In[16]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

def select_model(df, features):
    all_X = df[features]
    all_y = df["Survived"]
    
    models = [
        {
            "name": "LogisticRegression",
            "estimator": LogisticRegression(),
            "hyperparameters":
            {
                "solver": ["newton-cg", "lbfgs", "liblinear"]
            }        
        },
        {
            "name": "KNeighborsClassifier",
            "estimator": KNeighborsClassifier(),
            "hyperparameters":
            {
                "n_neighbors": range(1,20,2),
                "weights": ["distance", "uniform"],
                "algorithm": ["ball_tree", "kd_tree", "brute"],
                "p": [1,2]
            }  
        },
        {
            "name":"RandomForestClassifier",
            "estimator": RandomForestClassifier(random_state=1),
            "hyperparameters":
            {
                "n_estimators": [4, 6, 9],
                "criterion": ["entropy", "gini"],
                "max_depth": [2, 5, 10],
                "max_features": ["log2", "sqrt"],
                "min_samples_leaf": [1, 5, 8],
                "min_samples_split": [2, 3, 5]
            } 
        }
    ]
    
    for model in models:
        print(model["name"])
        print("-"*20)
        
        gs = GridSearchCV(model["estimator"],
                       param_grid = model["hyperparameters"],
                       cv=10)
        gs.fit(all_X, all_y)
        model["best_params"] = gs.best_params_
        model["best_score"] = gs.best_score_
        model["best_model"] = gs.best_estimator_
        
        print("Best Score: {}".format(model["best_score"]))
        print("Best Parameters: {}".format(model["best_params"]))
        
    return models

result = select_model(train, cols)
        


# # Submission to Kaggle

# In[19]:


def save_submission_file(model, cols, filename="submission.csv"):
    holdout_data = holdout[cols]
    predictions = model.predict(holdout_data)
    
    holdout_ids = holdout["PassengerId"]
    submission_df = {"PassengerId" : holdout_ids,
                    "Survived": predictions}
    submission = pd.DataFrame(submission_df)
    submission.to_csv(filename, index=False)

best_rf_model = result[2]["best_model"]
save_submission_file(best_rf_model, cols,filename="submission.csv")


# In[ ]:




