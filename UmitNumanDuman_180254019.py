import pandas as pd
import numpy as np

# machine learning

from sklearn.svm import SVC, LinearSVC


train_df = pd.read_csv("D:\Download\strain.csv") # load training data into a pandas dataframe
test_df = pd.read_csv("D:\Download\stest.csv") # load test data into a pandas dataframe
combine = [train_df, test_df] # combine both dataframes for future data wrangling

train_df.describe(include=['O']) # get summary statistics of categorical columns in training data
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False) # calculate the mean survival rate for each passenger class and sort them in descending order
train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False) # calculate the mean survival rate for each gender and sort them in descending order
train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False) # calculate the mean survival rate for each number of siblings/spouses aboard and sort them in descending order
train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False) # calculate the mean survival rate for each number of parents/children aboard and sort them in descending order




# Drop 'Ticket' and 'Cabin' columns from train and test dataframes
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)

# Combine the two dataframes into a list
combine = [train_df, test_df]

# Add 'Title' column to both dataframes by extracting the title from 'Name' column
for dataset in combine:
    dataset['Title'] = dataset['Name'].str.extract(" ([A-Za-z]+)\,", expand=False)

# Create a crosstab to show the relationship between 'Title' and 'Sex' in train dataframe
pd.crosstab(train_df['Title'], train_df['Sex'])

# Replace certain titles with 'Rare' and merge similar titles together
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', \
                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

# Create a mapping for 'Title' values to numerical values
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

# Map 'Title' values to numerical values and fill any missing values with 0
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

# Drop 'Name' and 'PassengerId' columns from train dataframe and 'Name' column from test dataframe
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)

# Combine the two dataframes into a list
combine = [train_df, test_df]

for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)

guess_ages = np.zeros((2, 3))

for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                               (dataset['Pclass'] == j + 1)]['Age'].dropna()


            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5

    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j + 1), \
                        'Age'] = guess_ages[i, j]

    dataset['Age'] = dataset['Age'].astype(int)

train_df.head()
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)

# Assign numerical values to age bands
def assign_age_band(age):
    if age <= 16:
        return 0
    elif age <= 32:
        return 1
    elif age <= 48:
        return 2
    elif age <= 64:
        return 3
    else:
        return 4  # for ages above 64

# Apply age banding to both train and test datasets
for dataset in combine:
    dataset['AgeBand'] = dataset['Age'].apply(assign_age_band)

# Drop the original Age feature and add the new AgeBand feature to the datasets
train_df = train_df.drop(['Age'], axis=1)
test_df = test_df.drop(['Age'], axis=1)
combine = [train_df, test_df]

# Create a new feature called FamilySize by combining the SibSp and Parch features
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

# Analyze the correlation between FamilySize and survival rate
train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived',
                                                                                                ascending=False)

# Create a new feature called IsAlone to indicate if a passenger is traveling alone or not
for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

# Drop the Parch, SibSp, and FamilySize features from the datasets
train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]

# Create a new feature called Age*Class by combining the Age and Pclass features
for dataset in combine:
    dataset['Age*Class'] = dataset.AgeBand * dataset.Pclass

# Find the most frequent port of embarkation and fill in missing values with it
freq_port = train_df.Embarked.dropna().mode()[0]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

# Analyze the correlation between Embarked and survival rate
train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived',
                                                                                            ascending=False)

# Map the Embarked feature to numerical values
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
test_df.head()
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand',
                                                                                            ascending=True)
for dataset in combine:
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

# Remove 'FareBand' column from train_df
train_df = train_df.drop(columns=['FareBand'])

# Combine train_df and test_df into a list called 'combine'
combine = [train_df, test_df]


# Split data into features and target
X_train = train_df.drop(columns=["Survived"])
Y_train = train_df["Survived"]

# Remove 'PassengerId' column from test_df and create a copy of the resulting DataFrame
X_test = test_df.drop(columns=["PassengerId"]).copy()

# I used svc method...

svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)




# Create a dataframe for the submission file
submission = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived": Y_pred
})

# Save the submission dataframe as a csv file
submission.to_csv('D:\Download\gender_submission.csv', index=False)
