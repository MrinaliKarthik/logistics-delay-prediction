import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


''' Pre-processing the data '''

#importing the data 
df=pd.read_csv('smart_logistics_dataset.csv')

#preprocessing data 
df.isnull().sum()

df=df.dropna() #dropping the null values 

df.isnull().sum()

#removing the redundant variables 
df=df.drop(columns=['Asset_ID'])

#extracting date from 'timestamp' column
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Date'] = df['Timestamp'].dt.date

#converting the date to months 
df['Month'] = df['Timestamp'].dt.month

#dropping the timestamp and date column 
df=df.drop(columns=['Timestamp', 'Date'])



'''Exploratory data analysis through data visualization '''

# 1. Distribution of the target variable (Logistics_Delay)
df['Logistics_Delay'].value_counts()

plt.figure(figsize=(12,10))
sns.countplot(data=df, x='Logistics_Delay', palette=['yellowgreen','firebrick'])
plt.title("Distribution of Logistics Delay", fontsize=20, fontweight='bold')
plt.xlabel("Logistics Delay (0 = On-Time, 1 = Delayed)", fontsize=18)
plt.ylabel("Number of Shipments", fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()

# 2. Delayed vs non-delayed shipments across months
#converting the numeric value in the month column to month names 
month_map = {1: "January", 2: "February", 3: "March", 4: "April",5: "May", 6: "June", 7: "July", 8: "August",9: "September", 10: "October", 11: "November", 12: "December"}
df['MonthName'] = df['Month'].map(month_map)

df['DelayStatus'] = df['Logistics_Delay'].map({0: "no delay", 1: "delay"})

month_order = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

plt.figure(figsize=(12,10))
sns.countplot(x='MonthName', hue='DelayStatus', data=df, palette=['salmon','deepskyblue'], order=month_order)
plt.title("Delayed vs Non-Delayed Shipments by Month", fontsize=20,fontweight='bold')
plt.xlabel("Month", fontsize=18)
plt.ylabel("Number of Shipments", fontsize=18)
plt.xticks(rotation=45,fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=18)
plt.show()

# 3. Grouped bar chart of Logistics_Delay counts segmented by Traffic_Status
plt.figure(figsize=(12,10))
sns.countplot(x='Traffic_Status' , hue='DelayStatus', data=df, palette=['forestgreen','tomato'], hue_order=["no delay", "delay"])
plt.title("Logistics Delay by Traffic Status", fontsize=20,fontweight='bold')
plt.xlabel("Traffic Status", fontsize=18)
plt.ylabel("Number of Shipments", fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(title="Delay Status",fontsize=18)
plt.show()

# 4. Box plot comparing Waiting_Time across the two Logistics_Delay classes
plt.figure(figsize=(12,10))
sns.boxplot(x='DelayStatus', y='Waiting_Time', data=df, palette=['seagreen','lightpink'], order=["no delay", "delay"])
plt.title("Waiting Time by Delay Status", fontsize=20,fontweight='bold')
plt.xlabel("Delay Status", fontsize=18)
plt.ylabel("Waiting Time", fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=18)
plt.show()


#dummifying the categorical variables and removing redundant variables for building a predictive model 
df=df.drop(columns=['MonthName','DelayStatus'])
df=pd.get_dummies(df, drop_first=True)

'''Building the models'''

#splitting the data into train and test set 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score

x=df.drop(columns=['Logistics_Delay'])
y=df['Logistics_Delay']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)

''' Feature Selection ''' 

#Initialize 
logmodel= LogisticRegression(solver='liblinear')

#Fitting the model 
logmodel.fit(x_train, y_train)

#Evaluating the model 
y_predict=logmodel.predict(x_test)
score=f1_score(y_test, y_predict)
print ("The F1 score is", score) #1

#Initializing feature selection 
sfs=SFS(logmodel, k_features=(1, 17), forward=True, scoring='f1', cv=5) 
sfs.fit(x_train,y_train)

#Features selected 
sfs.k_feature_names_

#Transforming the data with selected features 
x_train_sfs=sfs.transform(x_train)
x_test_sfs=sfs.transform(x_test)

#Fitting the model with new features 
logmodel.fit(x_train_sfs,y_train)

#Evaluating the model 
y_pred=logmodel.predict(x_test_sfs)
score1 = f1_score(y_test,y_pred) #1
print ("The F1 score is", score1)  

cm3=pd.DataFrame(confusion_matrix(y_test,y_pred,labels=[0,1]),index=["Actual:0", "Actual:1"],columns=["Pred:0",'Pred:1'])
print(cm3)

corr= df.corr() #correlation table 

#The reason behind the perfect f1 score above is the inclusion of highly correlated variable Traffic_Status_Heavy=+0.62
#Hence, to improve the predictive model, removal of this variable is important 

##removing Traffic_Status_Heavy and running feature selection to improve the model 

x=df.drop(columns=['Logistics_Delay', 'Traffic_Status_Heavy'])
y=df['Logistics_Delay']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1) 

#Initialize 
logmodel= LogisticRegression(solver='liblinear')

#Fitting the model 
logmodel.fit(x_train, y_train)

#Evaluating the model 
y_predict=logmodel.predict(x_test)
score=f1_score(y_test, y_predict)
print ("The F1 score is", score) #0.80

#Initializing feature selection 
sfs=SFS(logmodel, k_features=(1, 16), forward=True, scoring='f1', cv=5) 
sfs.fit(x_train,y_train)

#Features selected 
sfs.k_feature_names_

#Transforming the data with selected features 
x_train_sfs=sfs.transform(x_train)
x_test_sfs=sfs.transform(x_test)

#Fitting the model with new features 
logmodel.fit(x_train_sfs,y_train)

#Evaluating the model 
y_pred=logmodel.predict(x_test_sfs)
score1 = f1_score(y_test,y_pred) 
print ("The F1 score is", score1) #0.78

cm2=pd.DataFrame(confusion_matrix(y_test,y_pred,labels=[0,1]),index=["Actual:0", "Actual:1"],columns=["Pred:0",'Pred:1'])
print(cm2)




''' Logistic Regression '''

#Building model - Attemp 1 
#Building a logistic regression model with the variables chosen from feature selection 

x1=df[['Latitude','Temperature','Shipment_Status_Delivered','Shipment_Status_In Transit','Traffic_Status_Detour','Logistics_Delay_Reason_Traffic']]
y=df['Logistics_Delay']

x1_train,x1_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)

logmodel1= LogisticRegression(solver='liblinear') #initialize 

logmodel1.fit(x1_train,y_train) #train
logmodel1.intercept_
logmodel1.coef_

probabilities1=logmodel1.predict_proba(x1_test) #predict 
prediction1=logmodel1.predict(x1_test)

#Evaluating the model based on recall, precision, f1 scores and confusion matrix 

f1_score(y_test, prediction1) #0.80

recall_score(y_test, prediction1) #0.72

precision_score(y_test, prediction1) #0.91

con_max=pd.DataFrame(confusion_matrix(y_test,prediction1,labels=[0,1]),index=["Actual:0", "Actual:1"],columns=["Pred:0",'Pred:1'])
print(con_max) 

#Building model - Attempt 2 

x=df.drop(columns=['Logistics_Delay', 'Traffic_Status_Heavy'])
y=df['Logistics_Delay']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)

logmodel= LogisticRegression(solver='liblinear') #initialize 

logmodel.fit(x_train,y_train) #train
logmodel.intercept_
logmodel.coef_

probabilities=logmodel.predict_proba(x_test) #predict 
prediction=logmodel.predict(x_test)

#Evaluating the model based on recall, precision, f1 scores and confusion matrix 

f1_score(y_test, prediction) #0.80

recall_score(y_test, prediction) #0.71

precision_score(y_test, prediction) #0.91

cm_lr=pd.DataFrame(confusion_matrix(y_test,prediction,labels=[0,1]),index=["Actual:0", "Actual:1"],columns=["Pred:0",'Pred:1'])
print(cm_lr) 

#changing the threshold from 0.5 to 0.3
probabilities=logmodel.predict_proba(x_test)

y_probs=probabilities[:,1]

y_pred3=np.where(y_probs>0.3,1,0)

cm_lr1=pd.DataFrame(confusion_matrix(y_test,y_pred3,labels=[0,1]),index=["Actual:0", "Actual:1"],columns=["Pred:0",'Pred:1'])
print(cm_lr1) 

recall_score(y_test, y_pred3) #0.95 

f1_score(y_test, y_pred3) #0.87

precision_score(y_test, y_pred3) #0.8
 
#This model gives the least number of False Negative, so it makes more sense to go with this model 




'''Decision tree'''

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV

dt=DecisionTreeClassifier(random_state=1) #initialize

dt.fit(x_train,y_train) #train

dt.tree_.max_depth

#F1 score for training data 
tree_pred_train=dt.predict(x_train)
f1_score(y_train,tree_pred_train) #f1 = 1

tree_pred_test=dt.predict(x_test)
f1_score(y_test,tree_pred_test) #f1 = 0.80 

cm_dt1=pd.DataFrame(confusion_matrix(y_test,tree_pred_test,labels=[0,1]),index=["Actual:0", "Actual:1"],columns=["Pred:0",'Pred:1'])
print(cm_dt1)

#Tuning the hyper parameter 
parameter_grid={'max_depth':range(1,14),'min_samples_split':range(2,20)}

grid=GridSearchCV(dt,parameter_grid,verbose=3,scoring='f1',cv=5) #initialize 
grid.fit(x_train,y_train)

grid.best_params_ #best hyperparameters 
#{'max_depth': 14, 'min_samples_split': 2}

#building the model with tuned hyper parameters 
dt=DecisionTreeClassifier(max_depth=12, min_samples_split=2, random_state=1)
dt.fit(x_train,y_train)

#Training F1 score after tuning the model 
train_tuned=dt.predict(x_train)
f1_score(y_train,train_tuned) #f1 = 0.99 ; dropped from 1 to 0.99 

#Testing F1 score after tuning the model 
test_tuned=dt.predict(x_test)
f1_score(y_test,test_tuned) #f1 = 0.80 ; remained same 

#the decision tree is at optimal functioning hence no noticiable improvement in the f1 score in the train and test data 

cm_dt=pd.DataFrame(confusion_matrix(y_test,test_tuned,labels=[0,1]),index=["Actual:0", "Actual:1"],columns=["Pred:0",'Pred:1'])
print(cm_dt) 

recall_score(y_test,test_tuned) #0.74

precision_score(y_test,test_tuned) #0.86



'''Random Forest'''

from sklearn.ensemble import RandomForestClassifier

x=df.drop(columns=['Logistics_Delay', 'Traffic_Status_Heavy'])
y=df['Logistics_Delay']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)

rf=RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(x_train,y_train)

#Training F1 score 
train_rf=rf.predict(x_train)
print("Training F1 Score:", f1_score(y_train, train_rf)) #1.0

#Testing F1 score 
test_rf=rf.predict(x_test)
print("Training F1 Score:", f1_score(y_test, test_rf)) #0.82

cm_rf=pd.DataFrame(confusion_matrix(y_test,test_rf,labels=[0,1]),index=["Actual:0", "Actual:1"],columns=["Pred:0",'Pred:1'])
print(cm_rf) 

recall_score(y_test,test_rf) #0.75

precision_score(y_test,test_rf) #0.89
#Tuning hyperparameters for random forest 

param_grid = {'n_estimators': [50, 100, 200],'max_depth': [None, 5, 10, 20],'min_samples_split': [2, 5, 10],'min_samples_leaf': [1, 2, 4],'max_features': ['sqrt', 'log2']}

grid_search=GridSearchCV(estimator=rf,param_grid=param_grid,cv=5,scoring='f1',n_jobs=-1,verbose=1)

grid_search.fit(x_train,y_train)

#Best parameters and model 
print("Best Parameters:", grid_search.best_params_)
best_rf=grid_search.best_estimator_

#Evaluate on training set
train_pred_rfs=best_rf.predict(x_train)
print("Training F1 Score:", f1_score(y_train, train_pred_rfs)) #0.94

#Evaluate on testing set
test_pred_rfs=best_rf.predict(x_test)
print("Training F1 Score:", f1_score(y_test, test_pred_rfs)) #0.82

cm_rf1=pd.DataFrame(confusion_matrix(y_test,test_pred_rfs,labels=[0,1]),index=["Actual:0", "Actual:1"],columns=["Pred:0",'Pred:1'])
print(cm_rf1) 

recall_score(y_test,test_pred_rfs) #0.78

precision_score(y_test,test_pred_rfs) #0.87


'''After comparing all the three models, we choose to go with the logtistic regression model because it 
gives the least false negative cases'''
 

##Predicting logistic delays through logistic regression (80% accuracy)

logmodel.predict([[75.7743,108.942,747,25.5,75.9,21,315,3,83.4,248,4,0,1,0,1,0]]) #19th entry in df (1) wrong prediction 

logmodel.predict([[-19.2577,66.744,474,27.7,71.2,50,311,3,91.6,266,9,0,1,0,0,0]]) #36 entry in df (1) correct prediction 

logmodel.predict([[-15.6005, -92.1974,130,18.1,78.3,39,424,9,92.3,143,1,1,0,0,1,0]]) #51 entry in df (0) wrong prediction

logmodel.predict([[14.9079,50.1606,438,26.5,69.3,34,378,5,88.7,256,7,0,0,0,0,1]]) #67 entry in df (1) correct prediction 

logmodel.predict([[-53.3732,-99.637,186,18.9,70.5,38,344,1,95.6,252,8,0,1,0,1,0]]) #80 entry in df (0) correct prediction 

logmodel.predict([[83.2858,-66.1069,179,25,71.1,60,325,1,66.8,226,7,0,0,1,0,1]]) #120 entry in df (1) correct prediction 

logmodel.predict([[-32.724,82.9577,363,27,70.9,52,482,6,62.9,248,12,0,0,1,0,1]]) #144 entry in df (1) correct prediction

logmodel.predict([[40.5478,147.026,494,19.2,58.4,52,231,10,98.5,201,5,0,0,0,0,1]]) #193 entry in df (1) correct prediction 
 
logmodel.predict([[-55.5154,-163.694,392,27.4,69,31,347,9,98,133,12,1,0,1,0,1]]) #286 entry in df (0) correct prediction 

logmodel.predict([[59.8356,-114.42,157,18.7,73.6,55,323,9,81.3,141,9,0,0,0,0,0]]) #999 entry in df (1) correct prediction






