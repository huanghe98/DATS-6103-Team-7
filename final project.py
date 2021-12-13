# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%%

import time
start=time.perf_counter()

#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%%

def cleanDfLocation(row, colname):
  place = row[colname].strip()
  if('CA' in place or 'NV' in place or 'HI' in place): return 'West'
  if('WA' in place or 'OR' in place or 'ID' in place or 'MT' in place or 'WY' in place or 'AK' in place): return 'Northwest'
  if('UT' in place or 'AZ' in place or 'CO' in place or 'NM' in place or 'TX' in place or 'OK' in place): return 'Southwest'
  if('ND' in place or 'SD' in place or 'NE' in place or 'KS' in place or 'MN' in place or 'IA' in place): return 'Midwest'
  if('MO' in place or 'WI' in place or 'IL' in place or 'MI' in place or 'IN' in place or 'OH' in place): return 'Midwest'
  if('KY' in place): return 'Midwest'
  if('LA' in place or 'AR' in place or 'MS' in place or 'TN' in place or 'AL' in place or 'GA' in place): return 'Southeast'
  if('SC' in place or 'NC' in place or 'FL' in place): return 'Southeast'
  if('VA' in place or 'WV' in place or 'DC' in place or 'PA' in place or 'MD' in place or 'DE' in place): return 'Mid-Atlantic'
  if('NJ' in place or 'NY' in place): return 'Mid-Atlantic'
  if('VT' in place or 'NH' in place or 'CT' in place or 'RI' in place or 'MA' in place or 'ME' in place): return 'Northeast'
  else: return 'International'
  return np.nan

def cleanDfTitle(row,colname,posarray):
  pos=row[colname].strip()
  if('Manager' in pos): return 'Manager'
  if(pos==posarray[1] or pos==posarray[10] or pos==posarray[13]): return 'Engineer'
  if(pos==posarray[3] or pos==posarray[4] or pos==posarray[9]): return 'Analyst'
  if(pos==posarray[7] or pos==posarray[8] or pos==posarray[11]): return 'Sales'
  else: return 'Other'
  return np.nan

def cleanDfCompany(row,colname):
  comp=str(row[colname]).strip().lower()
  if('goog' in comp): return 'google'
  if('amzn' in comp or 'amazon' in comp): return 'amazon'
  if('microsoft' in comp or 'msft' in comp): return 'microsoft'
  if('facebook' in comp or 'fb' in comp): return 'facebook'
  if('appl' in comp): return 'apple'
  if('oracle' in comp or 'orcl' in comp): return 'oracle'
  if('salesforce' in comp or 'crm' in comp): return 'salesforce'
  if('intel' in comp or 'intc' in comp): return 'intel'
  if('cisco' in comp or 'csco' in comp): return 'cisco'
  if('ibm' in comp): return 'ibm'
  else: return comp
  return np.nan

#%%
df = pd.read_csv("salary.csv")

#%%

df['location'] = df.apply(cleanDfLocation, colname='location', axis=1)
posarray=df['title'].unique()
df['title'] = df.apply(cleanDfTitle, colname='title',posarray=posarray, axis=1)
df['company']=df.apply(cleanDfCompany,colname='company',axis=1)

#%%

# drop NA 
df_nona = df.dropna(subset=["Education"]) # drop NAs in Education
df_nona = df_nona.dropna(subset=["Race"]) # drop NAs in race
df_nona = df_nona.dropna(subset=["gender"]) # drop NAs in gender

#%%

# creat a new boolean variable as dependent variable for logistic regression
newdf = df_nona.iloc[:,[0,1,3,4,5,6,7,9,10,11,12,27,28]].copy() # pick up useful columns
newdf["stock"] = 0 # creat a new column
newdf["stock"][newdf.stockgrantvalue > 0] = 1 # person with granted stock is 1

#%%

# define a function to keep only year in timestamp
def cleanDfTime(row, colname):
  time = row[colname].strip()
  if('2021' in time): return '2021'
  if('2020' in time): return '2020'
  if('2019' in time): return '2019'
  if('2018' in time): return '2018'
  if('2017' in time): return '2017'
  else: return time
  
#%%
newdf['year'] = newdf.apply(cleanDfTime, colname='timestamp', axis=1)
newdf.head()
# Only observations in 2021 and 2020 has no NA in education, race and gender, so year only has two values(2020 and 2021)

#%%[markdown]
## PART 1. introduction and EDA

labels=['Manager','Engineer','Analyst','Sales','Other']
sizes=[len(df.loc[df.iloc[:,3]=='Manager']),len(df.loc[df.iloc[:,3]=='Engineer']),len(df.loc[df.iloc[:,3]=='Analyst']),len(df.loc[df.iloc[:,3]=='Sales']),len(df.loc[df.iloc[:,3]=='Other'])]
explode=[0.1,0,0,0,0]

fig1,ax1=plt.subplots()
ax1.pie(sizes,explode=explode,labels=labels,autopct='%.0f%%',startangle=360)
ax1.axis('equal')
plt.title('Distribution of Positions')

plt.show()

#%%

labels2=['West','Northwest','Southwest','Midwest','Southeast','Mid-Atlantic','Northeast','International']
sizes2=[len(df.loc[df.iloc[:,5]==labels2[0]]),len(df.loc[df.iloc[:,5]==labels2[1]]),len(df.loc[df.iloc[:,5]==labels2[2]]),len(df.loc[df.iloc[:,5]==labels2[3]]),len(df.loc[df.iloc[:,5]==labels2[4]]),len(df.loc[df.iloc[:,5]==labels2[5]]),len(df.loc[df.iloc[:,5]==labels2[6]]),len(df.loc[df.iloc[:,5]==labels2[7]])]
explode2=[0,0,0,0,0,0,0,0.1]

fig2,ax2=plt.subplots()
ax2.pie(sizes2,explode=explode2,labels=labels2,autopct='%.0f%%',startangle=90)
ax2.axis('equal')
plt.title('Distribution of Locations')

plt.show()

#%%

p=sns.boxplot(x="stockgrantvalue", y="totalyearlycompensation", data=df, showfliers=False)
p.set_xlabel('Stocks Granted',fontsize=16)
p.set_ylabel("Total Yearly Compensation",fontsize=16)
p.set_title('Income for Those Receiving and Not Receiving Stocks',fontsize=16)

#%%

titlar=['Manager','Sales','Engineer','Analyst','Other']
df2=df
df2['title']=pd.Categorical(df2['title'],titlar)

q=sns.boxplot(x="title", y="totalyearlycompensation", data=df2, showfliers=False)
q.set_xlabel('Position',fontsize=16)
q.set_ylabel("Total Yearly Compensation",fontsize=16)
q.set_title('Income for Different Positions',fontsize=16)

#%%

meanMan=df.loc[df.iloc[:,3]=='Manager']['totalyearlycompensation'].mean()
meanSal=df.loc[df.iloc[:,3]=='Sales']['totalyearlycompensation'].mean()
meanEng=df.loc[df.iloc[:,3]=='Engineer']['totalyearlycompensation'].mean()
meanAna=df.loc[df.iloc[:,3]=='Analyst']['totalyearlycompensation'].mean()
meanOth=df.loc[df.iloc[:,3]=='Other']['totalyearlycompensation'].mean()

#%%

meanWes=df.loc[df.iloc[:,5]=='West']['totalyearlycompensation'].mean()
meanNW=df.loc[df.iloc[:,5]=='Northwest']['totalyearlycompensation'].mean()
meanMid=df.loc[df.iloc[:,5]=='Mid-Atlantic']['totalyearlycompensation'].mean()
meanNE=df.loc[df.iloc[:,5]=='Northeast']['totalyearlycompensation'].mean()
meanSW=df.loc[df.iloc[:,5]=='Southwest']['totalyearlycompensation'].mean()
meanMW=df.loc[df.iloc[:,5]=='Midwest']['totalyearlycompensation'].mean()
meanSE=df.loc[df.iloc[:,5]=='Southeast']['totalyearlycompensation'].mean()
meanInt=df.loc[df.iloc[:,5]=='International']['totalyearlycompensation'].mean()

#%%

locar=['West','Northwest','Mid-Atlantic','Northeast','Southwest','Midwest','Southeast','International']
df3=df
df3['location']=pd.Categorical(df3['location'],locar)

r=sns.boxplot(x="location", y="totalyearlycompensation", data=df3, showfliers=False)
r.set_xlabel('Location',fontsize=16)
r.set_ylabel("Total Yearly Compensation",fontsize=16)
r.set_title('Income for Different Regions',fontsize=16)
plt.xticks(rotation=90)

#%%[markdown]
## PART 2. model building and evolution for total sallary

#%% 

# quickly check the simple model for total sallary
from statsmodels.formula.api import ols

linearmodel1 = ols(formula='totalyearlycompensation ~ yearsofexperience + yearsatcompany + C(gender) + C(Race) + C(Education) + C(year)', data = newdf)
linearmodel1_fit = linearmodel1.fit()
print(linearmodel1_fit.summary())


#%%[markdown]
## PART 3. model building and evolution for stock

# Stocks and stock options are a common and effective method for companies to attract employees and reduce employee turnover. For technical talents such as data scientists, companies are very willing to grant employees stocks to ensure their technological advantages. Most data scientists also are willing to get stocks.
#
# So, in this section, we want to find what factors affect whether data scientists will get stocks or stock options and use KNN model and other classifier models to help data scientists predict whether they can get stocks.


#%% model building and evolution for stock

# quickly check the simple model for stock
import statsmodels.api as sm
from statsmodels.formula.api import glm

glm1 = glm(formula='stock ~ totalyearlycompensation +yearsofexperience + yearsatcompany + C(gender) + C(Race) + C(Education) + C(year) +C(location)+C(title)', data=newdf, family=sm.families.Binomial())
glm1fit = glm1.fit()
print(glm1fit.summary())

#%%

# build a new model with all significant variables
glm2 = glm(formula='stock ~ totalyearlycompensation +yearsofexperience + yearsatcompany + C(Race) + C(Education)+ C(year) +C(title) ', data=newdf, family=sm.families.Binomial())
glm2fit = glm2.fit()
print(glm2fit.summary())

#%%

# prediction and confusion matrix
modelprediction = pd.DataFrame( columns=['prb'], data= glm2fit.predict(newdf))
modelprediction['glm2'] = np.where(modelprediction['prb'] > 0.5, 1, 0)
print(pd.crosstab(newdf.stock, modelprediction.glm2,
rownames=['Actual'], colnames=['Predicted'],
margins = True))

# Predicted     0      1    All
# Actual                       
# 0          3014   3317   6331
# 1          1769  13491  15260
# All        4783  16808  21591



#%%

# define a function to calcute accuracy, precision...
def confusionmatrix(cut_off):
  table = pd.crosstab(newdf.stock, np.where(modelprediction['prb'] > cut_off, 1, 0),rownames=['Actual'], colnames=['Predicted'],margins = True)
  TN = table.iloc[0,0]
  FP = table.iloc[0,1]
  FN = table.iloc[1,0]
  TP = table.iloc[1,1]
  accuracy = round((TP + TN) / table.iloc[2,2],4)
  precision = round(TP / (TP + FP),4)
  recall_rate = round(TP / (TP + FN),4)
  F1_score = round(2*TP/(2*TP + FP +FN),4)
  # print(table)
  # print("The total accuaracy of",cut_off,"cut-off model is",accuracy)
  # print("The precision of",cut_off,"cut-off model is",precision)
  # print("The recall rate of",cut_off,"cut-off model is",recall_rate)
  # print("The F1 score of",cut_off,"cut-off model is",F1_score)
  return table,accuracy,precision,recall_rate,F1_score

#%%
# define a function to show confusion matrix and accuracy...
def printconfusionmatrix(cut_off):
  newscore = confusionmatrix(cut_off)
  print('The confusion matrix:\n',newscore[0])
  print("The total accuaracy of",cut_off,"cut-off model is",newscore[1])
  print("The precision of",cut_off,"cut-off model is",newscore[2])
  print("The recall rate of",cut_off,"cut-off model is",newscore[3])
  print("The F1 score of",cut_off,"cut-off model is",newscore[4])
  return

#%%
printconfusionmatrix(0.5)

#%%

# define a function to show accuracy for different cut-offs
def scorestable(cut_offs):
  accuaracylist = []
  precisionlist = []
  recall_ratelist = []
  F1_scorelist = []
  for cut_off in cut_offs:
    newscores = confusionmatrix(cut_off)
    accuaracylist.append(newscores[1])
    precisionlist.append(newscores[2])
    recall_ratelist.append(newscores[3])
    F1_scorelist.append(newscores[4])
  scotable = pd.DataFrame({'cut_offs':cut_offs,'accuracy':accuaracylist,'precision':precisionlist,'recall_rate':recall_ratelist,'F1_score':F1_scorelist})
  return scotable
    
#%%
cut_offs = [0.3,0.35,0.4,0.45,0.5]
scorestable(cut_offs)

#%% draw for different cut-offs
cutofftable = scorestable(cut_offs)
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.plot(cutofftable.cut_offs,cutofftable.accuracy,label="Accuracy")
ax.plot(cutofftable.cut_offs,cutofftable.precision,label="Precision")
ax.plot(cutofftable.cut_offs,cutofftable.recall_rate,label="Recall_rate")
ax.plot(cutofftable.cut_offs,cutofftable.F1_score,label="F1-score")
ax.legend(loc='best')
ax.set_xlabel("cut-off")
ax.set_ylabel("score")
ax.set_ylim(0.5,1.05)
ax.set_title("classification report for different cut-offs")
plt.show()

# The best cut-off point is around 0.58

#%% classifiers

# First, spilt total data set to train and test set
df_x = newdf[['yearsofexperience','yearsatcompany','totalyearlycompensation',]]
df_y = newdf['stock']

# 4:1 train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(df_x, df_y, test_size=0.2, stratify=df_y,random_state=10)

#%% 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report

print("\nReady to continue.")

#%% SVC()
svc = SVC()
svc.fit(X_train,y_train)
print(f'svc train score:  {svc.score(X_train,y_train)}')
print(f'svc test score:  {svc.score(X_test,y_test)}')
print(confusion_matrix(y_test, svc.predict(X_test)))
print(classification_report(y_test, svc.predict(X_test)))

#%% LinearSVC()
linearSVC = LinearSVC()
linearSVC.fit(X_train,y_train)
print(f'svc(kernel="linear") train score:  {linearSVC.score(X_train,y_train)}')
print(f'svc(kernel="linear") test score:  {linearSVC.score(X_test,y_test)}')
print(confusion_matrix(y_test, linearSVC.predict(X_test)))
print(classification_report(y_test, linearSVC.predict(X_test)))

#%% KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
print(f'KNN train score:  {knn.score(X_train,y_train)}')
print(f'KNN test score:  {knn.score(X_test,y_test)}')
print(confusion_matrix(y_test, knn.predict(X_test)))
print(classification_report(y_test, knn.predict(X_test)))

#%% use different n-neighbors to KNN 

# First calculate the training score and testing score
Klist = []
KNNtrainscore = []
KNNtextscore = []
for i in range(13):
  k = 2*i + 1
  Klist.append(k)
  knn = KNeighborsClassifier(n_neighbors=k)
  knn.fit(X_train,y_train)
  KNNtrainscore.append(round(knn.score(X_train,y_train),5))
  KNNtextscore.append(round(knn.score(X_test,y_test),5))

# Then draw the plot
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.plot(Klist,KNNtrainscore,label="training score")
ax.plot(Klist,KNNtextscore,label="testing score")
ax.legend(loc='best')
ax.set_xlabel("K")
ax.set_ylabel("score")
ax.set_ylim(0,1.05)
ax.set_title("KNeighborsClassifier")
plt.show()

#%% DecisionTreeClassifier()
dtree = DecisionTreeClassifier(max_depth=5, random_state=1)
# Fit dt to the training set
dtree.fit(X_train,y_train)
# Predict test set labels
y_test_pred = dtree.predict(X_test)
# Evaluate test-set accuracy
print(accuracy_score(y_test, y_test_pred))
print(confusion_matrix(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))

#%% ROC AUC
from sklearn.metrics import roc_auc_score, roc_curve
logreg = LogisticRegression()
logreg.fit(X_train,y_train)

# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(y_test))]
# predict probabilities
lr_probs = logreg.predict_proba(X_test)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# calculate scores
ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, lr_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc)) 
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
# plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
# show the plot
plt.show()

# The AUC value is 0.808, which means logistic model is a good prediction model.

#%%[markdown]
# To sum up, after checking the logistic regression model, KNN models, SVC model, SVC with linear kernel model, and decision tree model, we compared their performance and time cost. The best prediction method in these methods is KNN model, and the appropriate range of K value is between 9 and 15. In this range, the total accuracy of KNN models is around 0.78 and takes less time.

#%%[markdown]
## Conclusion
# In this project, we focused on our SMART questions and found some interesting inspirations for data scientists.
#
# We found the top 5 companies with best average salaries, which includes Netflix, Snap, Stripe, Lyft, Pinterest. Also, we found the top 5 cities with the best average salaries, which includes Los Gatos, Menlo Park, Mountain View, Cupertino, and Kirkland. So, if data scientists want to get a higher salary, these companies and cities are the best choice.
#
# For the first SMART question about what factors affect the salary level of data scientists, we found some related variables. An additional year of experience increased the base salary by around 4,281 dollars; years at the company actually had a negative impact on salary (although minimal); education seemed to have the biggest impact on base salary and master’s degree increased the base salary by around 14,528 dollars.
#
# For the second SMART question on how to predict who was granted stocks, we found that years at company and years of experience all have a negative impact on stock. And when it comes to prediction, KNN model is the best method for this data set and the appropriate K values range from 9 to 15.
#
# However, overall, our current research is not deep enough. For follow-up research, maybe we can use the PCA method to help us determine the importance of variables and reduce the number of variables in the linear regression model. We also can try different calculation methods in each classifier model.


# %%

print("Run time:",time.perf_counter()-start)

# %%

# %%
