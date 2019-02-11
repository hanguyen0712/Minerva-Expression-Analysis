#import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

#read csv file
df = pd.read_csv("D:/2018 internship - Minerva/expression_maturity_calc.csv")

#delete applicants with outstanding completion time
df0 = df.drop(df[df.time_sec > 80000].index)
#replace blankspace with 0
df0 = df0.replace(r'^\s+$', np.nan, regex=True)
df0 = df0.fillna(0)
#summary statistics
df0.info()
df0.describe()


#checkbox likelihood by number and by percentage
for col in df0.columns[18:]:
    df0[col].value_counts()
    df0[col].value_counts(1)

#Scatterplot matrix - overview about the correlation between variables
sns.set(style="ticks")
sns.pairplot(df0, hue="Reasonable responsibility")

#Correlation between Written and Expression Responsiveness
df0['written_responsiveness'].corr(df0['expression_responsiveness'])
#Scatterplot
sns.regplot(x=df0['written_responsiveness'], y=df0['expression_responsiveness'])


#Correlation between Written and Expression Openness
df0['written_openness'].corr(df0['expression_openness'])
#Scatterplot
sns.regplot(x=df0['written_openness'], y=df0['expression_openness'])


#Correlation between Written and Expression Responsibility
df0['written_responsibility'].corr(df0['expression_responsibility'])
#Scatterplot
sns.regplot(x=df0['written_responsibility'], y=df0['expression_responsibility'])


#Correlation between Written and Expression Empathy
df0['written_empathy'].corr(df0['expression_empathy'])
#Scatterplot
sns.regplot(x=df0['written_empathy'], y=df0['expression_empathy'])


#Correlation between Written and Expression Factors
df0['written_factors'].corr(df0['expression_factors'])
#Scatterplot
sns.regplot(x=df0['written_factors'], y=df0['expression_factors'])


#Maturity score by prompt
ms = pd.DataFrame()
ms['prompt'] = df0['prompt'].unique()
ms['mean_written_confidence'] = df0.prompt.map(df0.groupby(['prompt']).written_confidence.mean())
ms['mean_written_responsiveness'] = df0.prompt.map(df0.groupby(['prompt']).written_responsiveness.mean())
ms['mean_written_empathy'] = df0.prompt.map(df0.groupby(['prompt']).written_empathy.mean())
ms['mean_written_openness'] = df0.prompt.map(df0.groupby(['prompt']).written_openness.mean())
ms['mean_written_factors'] = df0.prompt.map(df0.groupby(['prompt']).written_factors.mean())
ms['mean_written_responsibility'] = df0.prompt.map(df0.groupby(['prompt']).written_responsibility.mean())
ms['mean_written_morality'] = df0.prompt.map(df0.groupby(['prompt']).written_morality.mean())
ms['mean_expression_responsiveness'] = df0.prompt.map(df0.groupby(['prompt']).expression_responsiveness.mean())
ms['mean_expression_empathy'] = df0.prompt.map(df0.groupby(['prompt']).expression_empathy.mean())
ms['mean_expression_openness'] = df0.prompt.map(df0.groupby(['prompt']).expression_openness.mean())
ms['mean_expression_factors'] = df0.prompt.map(df0.groupby(['prompt']).expression_factors.mean())
ms['mean_expression_responsibility'] = df0.prompt.map(df0.groupby(['prompt']).expression_responsibility.mean())


#Maturity by applicant
#group items belonging to an applicants
df2 = df0.groupby('app_id').agg(lambda x: x.tolist())

#delete entries that don't have 3 or 5 graders
df1 = pd.DataFrame(columns = ['prompt', 'State what they learned','Explain what they could have done differently','Describe development as a result of the experience','Reasonable responsibility'])
for i in range(len(df2['prompt'])):
  if len(df2['prompt'][i]) == 3 or len(df2['prompt'][i]) == 5:
        df1 = df1.append(df[i:i+1])

#return a number for prompt
for i in range(len(df1['prompt'])):
    df1['prompt'][i] = df1['prompt'][i][0]

#count the number of applicants in a prompt
promptDict = {}
for i in range(3,13):
    promptDict[i] = (df1['prompt'] == i).sum()
total = sum(promptDict.values())#total number of applicants

#check if list is homogeneous
def checkEqual(array):
    if len(set(array)) == 1:
        return True

#count how many lists are homogeneous in each column
for col in df1[1:]:
    counter = 0
    for i in df1[col]:
        if type(i) == list and checkEqual(i) == True:
            counter += 1
    print(counter, col ) #how many applicants whose APs are in agreement regarding the score
    print(round((counter/total)*100,2), col) #percentage of agreement

#prompt-based
for col in df1[1:]:
    for j in range(3, 13):
        counter = 0
        for i in range(len(df1['prompt'])):
            if df1['prompt'][i] == j and type(df1[col][i]) == list and checkEqual(df1[col][i]) == True:
                counter += 1
        print("For prompt ", j , " there are ", counter, " APs in agreement regarding ", col)
        print(round((counter/promptDict[j])*100,2)) #percentage of agreement


#linear regression to predict expression_responsibility from written responsibility
#define predictor and outcome
X = df0[["written_responsibility", "time_sec"]]
y = df0["expression_responsibility"]
#split X and y into training set and test set (test set = 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
#reshape the data
X_train = X_train.values.reshape((4276, 2))
X_test = X_test.values.reshape((1833, 2))
#check the size of training and test sets
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)

#linear regression model
lm = linear_model.LinearRegression()
#train the model
model = lm.fit(X_train, y_train)
pred = lm.predict(X_test)
#intercept
lm.intercept_
#coefficients
lm.coef_
#accuracy score
print ("Score:", model.score(X_test, y_test))

#Scatterplot for true and predicted values
plt.scatter(y_test, pred)
plt.xlabel("True Values")
plt.ylabel("Predictions")

#Residual errors in training data
plt.scatter(model.predict(X_train), model.predict(X_train) - y_train, color = "green", s = 10, label = 'Train data')
#Residual errors in test data
plt.scatter(model.predict(X_test), model.predict(X_test) - y_test, color = "blue", s = 10, label = 'Test data')
#Line for zero residual error
plt.hlines(y = 0, xmin = 1.4, xmax = 2.2, linewidth = 0.1)
#Legend
plt.legend(loc = 'upper right')
#Title
plt.title("Residual errors")
#Show plot
plt.show()
