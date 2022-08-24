#!/usr/bin/env python
# coding: utf-8

# In[148]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn import metrics
from sklearn.metrics import confusion_matrix,recall_score,accuracy_score,roc_curve,roc_auc_score


# In[46]:


df = pd.read_csv('telecom_customer_churn.csv')


# # EDA
# ### Analyzing Dataset

# In[47]:


df.info()


# In[48]:


df.head()


# ## Checking Outliers of variables

# In[49]:


df_num = df[['Total Revenue','Tenure in Months','Monthly Charge','Avg Monthly Long Distance Charges','Avg Monthly GB Download','Total Charges']]

for var, predictor in enumerate(df_num):
    plt.figure(var)
    ax = sns.boxplot(data=df, x=predictor)


# ## Imputing Numerical Variables with Median value and Categorial Variables with Mode value. 

# In[50]:


df['Avg Monthly Long Distance Charges'].fillna(df['Avg Monthly Long Distance Charges'].median(), inplace=True)
df['Avg Monthly GB Download'].fillna(df['Avg Monthly GB Download'].median(), inplace=True)


# In[51]:


df = df.fillna(df.mode().iloc[0])


# In[52]:


df.isna().sum()


# ### Plotting Customer Status Against Independent Categorical Variables

# In[53]:


df_fig = df[['Total Revenue','Tenure in Months','Monthly Charge','Total Charges','Age']]

sns.set(rc={'figure.figsize':(5,5)})
for i, predictor in enumerate(df_fig):
    plt.figure(i)
    ax = sns.boxplot(data=df, x='Customer Status', y=predictor)


# ### Plotting Customer Status Against Independent Categorical Variables

# In[54]:


df_cat = df[['Gender', 'Married','Multiple Lines','Internet Service','Internet Type','Streaming TV','Streaming Movies','Streaming Music',                    
       'Unlimited Data','Contract']]
for i, predictor in enumerate(df_cat):
    plt.figure(i)
    ax = sns.countplot(data=df, y=predictor, hue= 'Customer Status',orient = 'h')
    ax.set(xlabel = None, ylabel = None)
    plt.title(str(predictor), loc='center')


# In[55]:


df_new = {"Customer Status": {"Churned": 1, "Stayed": 0, "Joined": 0}}
df = df.replace(df_new)


# In[56]:


df.head()


# ### Checking Correlation between Target Variable and Independent Numerical Variables 

# In[57]:


corr = df.corr()
corr.style.background_gradient(cmap='BuPu')


# ### Imputing Categorical Boolean Variables with Numerical Boolean Variables

# In[60]:


df_new1 = {"Gender": {"Male": 1, "Female": 0},"Married":{"Yes":1,"No":0},'Phone Service':{"Yes":1,"No":0},'Multiple Lines':{"Yes":1,"No":0},'Internet Service':{"Yes":1,"No":0},'Offer':{"Yes":1,"No":0},'Device Protection Plan':{"Yes":1,"No":0},'Premium Tech Support':{"Yes":1,"No":0},'Streaming TV':{"Yes":1,"No":0},'Streaming Movies':{"Yes":1,"No":0},"Contract":{"Month-to-Month":1,"One Year":2,"Two Year":3},'Streaming Music':{"Yes":1,"No":0},'Unlimited Data':{"Yes":1,"No":0},'Contract':{"Yes":1,"No":0}}
df = df.replace(df_new1)


# In[72]:


df_new2 = {"Contract": {"Month-to-Month":1,"One Year":2,"Two Year":3},"Churn Category":{'Competitor':1, 'Dissatisfaction':2, 'Other':3, 'Price':4, 'Attitude':5}}
df = df.replace(df_new2)


# In[76]:


df.head()


# ### Dropping of insignificant Variables

# In[82]:


df_test = df.drop(['Customer ID','Zip Code','City','Latitude','Longitude','Number of Referrals','Payment Method','Number of Dependents','Internet Type','Offer','Online Security','Online Backup','Device Protection Plan','Premium Tech Support','Paperless Billing','Total Refunds','Churn Reason'], axis = 1)


# # Data Modeling
# ### Splitting Dataset into Training and Testing subsets

# In[83]:


x=df_test.drop('Customer Status',axis=1)
y=df_test['Customer Status']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


#  ### Logistic Regression

# In[106]:


lr_model = LogisticRegression(class_weight='balanced', random_state=1)
lr_model.fit(x_train, y_train)
y_pred = lr_model.predict(x_test)
#acc_score, prec_score, rec_score = evaluate_model(y_test, y_pred)
print('Accuracy:', accuracy_score(y_test,y_pred))
print('Recall:', recall_score(y_test,y_pred))


# In[108]:


cm = confusion_matrix(y_test, y_pred)
ax = sns.heatmap(cm,annot=True,cmap='Purples')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');
plt.show()
print(cm)


# ### Decision Tree

# In[124]:


dt = tree.DecisionTreeClassifier()
dt.fit(x_train, y_train)

y_pred = dt.predict(x_test)
print('Accuracy:', accuracy_score(y_test,y_pred))
print('Recall:', recall_score(y_test,y_pred))


# In[125]:


cm = confusion_matrix(y_test, y_pred)
ax = sns.heatmap(cm,annot=True,cmap='Purples')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');
plt.show()
print(cm)


# ### Random Forest Classifier

# In[128]:


frst = RandomForestClassifier(n_estimators=200,random_state=1)
frst.fit(x_train, y_train)
y_pred=frst.predict(x_test)
print('Accuracy:', metrics.accuracy_score(y_test, y_pred))
print('Recall:', recall_score(y_test,y_pred))


# In[129]:


cm = confusion_matrix(y_test, y_pred)
ax = sns.heatmap(cm,annot=True,cmap='Purples')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');
plt.show()
print(cm)


# ## Plotting of ROC curve for Random Forest Classifier 

# In[163]:


fpr, tpr, _ = roc_curve(y_test, y_pred)
aucval = roc_auc_score(y_test, y_pred)
plt.figure(figsize=(10,8))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_Nb, tpr_Nb,"r",linewidth = 3)
plt.grid()
plt.xlabel ("1-Specificity")
plt.ylabel ("Sensitivity")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.title ("Random Forest ROC Curve")
plt.text(0.15,0.9,"AUC = "+str(round(aucval,4)))
plt.show()


# # Conclusion
# 
# #### Random Forest Classifier technique is the best model as it gives us the most accuracy of 91% and an AUC score of 0.8617.
# #### According to Correlation Matrix, Total Revenue and Total Charges variables are positively correlated with Tenure in Months variables.
# #### Also, as expected, Total Charges Variable is positively correlated with Total Revenue.
# #### Following are the Insights from EDA - 
# ##### 1. Unmarried Customers are more prone to churning. 
# ##### 2. Customers who have taken the Internet service have less churn rate compared to people who don't have internet service from this provider. 
# ##### 3. Also, among the customers who have the Internet service, customers who have Fiber Optic types of Internet are more likely to churn.
# ##### 4. Another Interesting insight is that Customers who have unlimited data have more churn rate than those who don't.
# ##### 5. Customers who have not committed to long-term plans (monthly plan) are more likely to churn compared to customers who have a one-year or two-year plan. 
