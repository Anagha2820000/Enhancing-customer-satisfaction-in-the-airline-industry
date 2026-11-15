#!/usr/bin/env python
# coding: utf-8

# ### Capstone Project
# ## Enhancing customer satisfaction in the airline industry: A comprehensive analysis

# In[1]:


#Importing the libraries
import warnings
warnings.filterwarnings('ignore')
import pandas as pd 
import pandas_profiling
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# In[2]:


#Read the data set
data= pd.read_csv(r"C:\Users\anagha\Desktop\capstone\Airline+Passenger+Satisfaction (1)\airline_passenger_satisfaction.csv")
data.head()


# In[3]:


#Basic informations about the dataset
print("Shape of the data: ", data.shape)


# In[4]:


data.dtypes


# In[5]:


data_cols= data.columns.tolist()
data_cols


# In[10]:


data.describe().T


# In[6]:


data.drop(columns=['ID'], inplace=True)


# In[7]:


def unique_values_names(df):
    for column in df:
        if df[column].dtype=='object':
            print(f'{column}:{df[column].unique()}')
            
unique_values_names(data)


# In[11]:


#Creating a profile report
profile=data.profile_report(title='Pandas Profiling')
profile


# In[8]:


fig = plt.figure(figsize=(22,6))
plt.suptitle("Target Distribution", fontsize=24, weight='bold')
grid= gridspec.GridSpec(nrows=1, ncols=2, figure= fig)
#Define the subplots 
ax1=  fig.add_subplot(grid[0,0])
ax1.set_title("Satisfaction count")
#With the help of seaborn we are drawing the count plot 

sns.countplot(x='Satisfaction', data=data, hue='Satisfaction',ax=ax1)
#To remove the border of the plot
for spine in ax1.spines.values():
    spine.set_visible(False)

#Second subplot 
ax2= fig.add_subplot(grid[0,1])
ax2.set_title("Target Weight")
label=list(data['Satisfaction'].value_counts().index)
value =list(data['Satisfaction'].value_counts().values)

#piechart
ax2.pie(value, labels=label, autopct='%1.1f%%',explode=(0,0.2))
plt.show()


# ### Data cleaning

# #### Handling the null values

# In[9]:


data.isnull().sum()


# #### Handling the null values 

# In[3]:


#Simple imputation
from sklearn.impute import SimpleImputer
fea_transformer = SimpleImputer(strategy="median")
data['Arrival Delay'] = fea_transformer.fit_transform(data[["Arrival Delay"]])


# #### Outlier handling

# In[11]:


exclude_=['Arrival Delay', 'Departure Delay','Age','Flight Distance', 'Departure and Arrival Time Convenience']
figure, axes = plt.subplots(len(exclude_), 2, figsize=(10,4*len(exclude_)))
axes = axes.flatten()
for i in range(len(exclude_)):
    sns.histplot(data[exclude_[i]], kde=True, ax=axes[i*2], bins=50)
    sns.boxplot(data[exclude_[i]], ax=axes[i*2+1])
    axes[i*2].set_title(exclude_[i])
    axes[i*2+1].set_title(exclude_[i])
    axes[i*2].set_xlabel('')
    axes[i*2+1].set_xlabel('')
    axes[i*2].set_ylabel('')
    axes[i*2+1].set_ylabel('')
plt.tight_layout()
plt.show()


# ##### IQR based outlier detection

# In[4]:


#Arrival and Departure delay are weakly correlated with the target variable, so we ignoring that variables 
sns.boxplot(data['Flight Distance'])


# In[5]:


#Finding the quartiles 
percentile25 = data['Flight Distance'].quantile(0.25)
percentile75 = data['Flight Distance'].quantile(0.75)

#Finding the upper and lower limits
iqr= percentile75-percentile25
upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr
print(upper_limit)
print(lower_limit)


# In[6]:


#Trimming outliers
new_df = data[(data['Flight Distance'] < 3739) & (data['Flight Distance'] > -1581)]
new_df.shape


# In[7]:


import matplotlib.pyplot as plt
plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.distplot(data['Flight Distance'])
plt.subplot(2,2,2)
sns.boxplot(data['Flight Distance'])
plt.subplot(2,2,3)
sns.distplot(new_df['Flight Distance'])
plt.subplot(2,2,4)
sns.boxplot(new_df['Flight Distance'])
plt.show()


# In[8]:


#Capping
import numpy as np
new_df_cap = data.copy()
new_df_cap['Flight Distance'] = np.where(
    new_df_cap['Flight Distance'] > upper_limit,
    upper_limit,
    np.where(
        new_df_cap['Flight Distance'] < lower_limit,
        lower_limit,
        new_df_cap['Flight Distance']))


# In[9]:


#Compare the plots after capping
plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.distplot(data['Flight Distance'])
plt.subplot(2,2,2)
sns.boxplot(data['Flight Distance'])
plt.subplot(2,2,3)
sns.distplot(new_df_cap['Flight Distance'])
plt.subplot(2,2,4)
sns.boxplot(new_df_cap['Flight Distance'])
plt.show()


# In[10]:


#Converting age into age groups
data= new_df_cap
# Define age bins and labels
bins = [0, 18, 35, 50, float('inf')]  # Specify your own age bins
labels = ['0-18', '19-35', '36-50', '51+']
data['Age'] = pd.cut(data['Age'], bins=bins, labels=labels, right=False)


# In[11]:


#Age group analysis
fig = plt.figure(figsize=(10,4))
grid= gridspec.GridSpec(nrows=1, ncols=2, figure= fig)
ax1=  fig.add_subplot(grid[0,0])
ax1.set_title("Age Group Count")
sns.countplot(x='Age', data=data, hue='Age',ax=ax1)
for spine in ax1.spines.values():
    spine.set_visible(False)


# #### Label Encoding

# In[11]:


from sklearn.preprocessing import LabelEncoder
categorical_columns = ['Gender','Age', 'Customer Type','Type of Travel','Class', 'Satisfaction',]
label_encoders = {}
for col in categorical_columns:
    label_encoders[col] = LabelEncoder()
    data[col] = label_encoders[col].fit_transform(data[col])


# In[18]:


#Arrival Delay and Departure Delay
sns.scatterplot(data=data, x='Arrival Delay', y='Departure Delay')
plt.tight_layout()
plt.show()


# In[20]:


#The figure shows that it is linearly related


# In[19]:


#Flight Distance and Arrival Delay
sns.scatterplot(data=data,x='Flight Distance', y='Arrival Delay')
plt.tight_layout()
plt.show()


# In[33]:


# There is no particular relationship between flight distance and arrival delay


# In[20]:


#Age and Satisfaction
sns.kdeplot(data=data, x='Age', hue='Satisfaction', fill=True)


# In[21]:


#Gender
busnes = data['Gender'].value_counts(normalize=True)
plt.figure(figsize=(6,4))
plt.pie(busnes, labels=busnes.index, autopct='%1.1f%%', colors=["pink","brown"])
plt.title('Gender')
plt.axis('equal')  
plt.show() 


# In[22]:


sns.countplot(data=data, x='Gender', hue='Satisfaction')


# In[23]:


sns.barplot(data=data, x='Seat Comfort', y='Flight Distance', hue='Satisfaction')


# In[24]:


sns.barplot(data=data, x='In-flight Entertainment', y='Flight Distance', hue='Satisfaction')


# In[25]:


sns.barplot(data=data, x='Leg Room Service', y='Flight Distance', hue='Satisfaction')


# In[26]:


sns.barplot(data=data, x='Class', y='Flight Distance', hue='Satisfaction')


# In[27]:


busnes = data['Customer Type'].value_counts(normalize=True)
plt.figure(figsize=(6,4))
plt.pie(busnes, labels=busnes.index, autopct='%1.1f%%')
plt.title('Distribution of Type of Travel')
plt.axis('equal')  
plt.show() 


# In[28]:


plt.figure(figsize=(6,4))
sns.histplot(data=data,x='Customer Type',hue='Customer Type',zorder=5,alpha=1,palette=['#FF5733', '#33B8FF'])
plt.title('Satisfaction  by Customer Type', fontsize=20, color='#a5a7ab')
plt.xlabel('Customer Type', fontsize=10)
plt.ylabel('Satisfaction', fontsize=10)
plt.xticks(fontsize=10, color='#8a8d93')
plt.yticks(fontsize=10, color='#8a8d93')
plt.show()


# In[29]:


busnes = data['Type of Travel'].value_counts(normalize=True)
plt.figure(figsize=(6,4))
plt.pie(busnes, labels=busnes.index, autopct='%1.1f%%')
plt.title('Distribution of Type of Travel')
plt.axis('equal')  
plt.show() 


# In[30]:


Classs = data['Class'].value_counts(normalize=True)
plt.figure(figsize=(6,4))
plt.pie(Classs, labels=Classs.index, autopct='%1.1f%%')
plt.title('Distribution of Class')
plt.axis('equal')  
plt.show() 


# In[31]:


plt.figure(figsize=(6,4))
sns.histplot(data=data,x='Class',hue='Class',zorder=5,alpha=1,palette='Set1')
plt.title('Satisfaction  by Class', fontsize=15, color='#a5a7ab')
plt.xlabel('Class', fontsize=10)
plt.ylabel('Satisfaction', fontsize=10)
plt.xticks(fontsize=10, color='#8a8d93')
plt.yticks(fontsize=10, color='#8a8d93')
plt.show()


# In[32]:


plt.figure(figsize=(7,4))
sns.histplot(data=data,x='In-flight Entertainment',hue='In-flight Entertainment',zorder=5,alpha=1,palette='Set1')
plt.title('Satisfaction  by Inflight entertainment', fontsize=10, color='#a5a7ab')
plt.xlabel('Inflight entertainment', fontsize=16)
plt.ylabel('Satisfaction', fontsize=16)
plt.xticks(fontsize=14, color='#8a8d93')
plt.yticks(fontsize=14, color='#8a8d93')
plt.show()


# In[33]:


plt.figure(figsize=(7,4))
sns.histplot(data=data,x='Type of Travel',hue='Type of Travel',zorder=5,alpha=1,palette=['#FF5733', '#33B8FF'])
plt.title('Satisfaction  by Type of Travel', fontsize=15, color='#a5a7ab')
plt.xlabel('Type of Travel', fontsize=10)
plt.ylabel('Satisfaction', fontsize=10)
plt.xticks(fontsize=10, color='#8a8d93')
plt.yticks(fontsize=10, color='#8a8d93')
plt.show()


# In[34]:


class_distribution = data.groupby(['In-flight Entertainment', 'Class']).size().reset_index(name='Count')

plt.figure(figsize=(6,4))

sns.barplot(data=class_distribution, x='In-flight Entertainment', y='Count', hue='Class', palette='Set1')

plt.title('Distribution of Travel Class by Inflight Entertainment', fontsize=15, fontfamily="Muli, sans-serif")
plt.xlabel('Inflight Entertainment', fontsize=16)
plt.ylabel('Count', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.legend(title='Class', title_fontsize='14')

plt.show()


# ### Feature Selection

# In[23]:


features=['Gender','Age','Customer Type','Type of Travel','Class','Flight Distance','Departure Delay','Arrival Delay',
 'Departure and Arrival Time Convenience','Ease of Online Booking','Check-in Service','Online Boarding','Gate Location',
 'On-board Service','Seat Comfort','Leg Room Service','Cleanliness','Food and Drink','In-flight Service',
   'In-flight Wifi Service','In-flight Entertainment','Baggage Handling']
X=data[features]
Y=data['Satisfaction']


# In[51]:


#Feature Selection using random forest
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

model = RandomForestClassifier(n_estimators=340)
model.fit(X, Y)

importances = model.feature_importances_

final_df = pd.DataFrame({"Features": pd.DataFrame(X).columns, "Importances": importances})
final_df.set_index('Importances')

final_df = final_df.sort_values('Importances', ascending=False)

threshold = 0.02  
selected_features = final_df.loc[final_df['Importances'] > threshold, 'Features'].tolist()

print("Selected Features:", selected_features)


# In[52]:


from eli5.sklearn import PermutationImportance
from sklearn.ensemble import RandomForestClassifier
import eli5

# Assuming X contains your feature matrix and y contains your target variable

# Train a RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(X, Y)

# Compute permutation importance
perm = PermutationImportance(rf, random_state=1).fit(X, Y)

# Display feature importance
eli5.show_weights(perm, feature_names=X.columns.tolist())


# In[53]:


print(len(selected_features))


# ### Machine learning models

# In[24]:


#Selected features after feature selection
features=['Online Boarding', 'In-flight Wifi Service', 'Class', 'Type of Travel', 'In-flight Entertainment', 'Seat Comfort', 
       'Flight Distance', 'Leg Room Service', 'Customer Type', 'Ease of Online Booking', 'On-board Service', 'Cleanliness', 
       'Check-in Service', 'In-flight Service', 'Baggage Handling']
X=data[features]

Y=data['Satisfaction']


# In[25]:


# Splitting X and Y
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 42, stratify = data['Satisfaction'] )


# In[56]:


# Checking dimensions
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Y_train shape:", Y_train.shape)
print("Y_test shape:", Y_test.shape)


# In[57]:


# Logistic Regression Algorithm
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(random_state = 42)
logreg.fit(X_train, Y_train)


# In[58]:


# Plotting a graph for n_neighbors 
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

X_axis = list(range(1, 50))
acc = pd.Series()
x = range(1,50)

for i in list(range(1, 50)):
    knn_model = KNeighborsClassifier(n_neighbors = i) 
    knn_model.fit(X_train, Y_train)
    prediction = knn_model.predict(X_test)
    acc = acc.append(pd.Series(metrics.accuracy_score(prediction, Y_test)))
plt.plot(X_axis, acc)
plt.xticks(x)
plt.title("Finding best value for n_estimators")
plt.xlabel("n_estimators")
plt.ylabel("Accuracy")
plt.grid()
plt.show()
print('Highest value: ',acc.values.max())


# In[59]:


# K nearest neighbors Algorithm
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 7, metric = 'minkowski', p = 2)
knn.fit(X_train, Y_train)


# In[60]:


# Naive Bayes Algorithm
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, Y_train)


# In[61]:


# Decision tree Algorithm
from sklearn.tree import DecisionTreeClassifier
dectree = DecisionTreeClassifier(criterion = 'entropy', random_state = 42)
dectree.fit(X_train, Y_train)


# In[62]:


# Random forest Algorithm
from sklearn.ensemble import RandomForestClassifier
ranfor = RandomForestClassifier(n_estimators = 11, criterion = 'entropy', random_state = 42)
ranfor.fit(X_train, Y_train)


# In[63]:


# Making predictions on test dataset
Y_pred_logreg = logreg.predict(X_test)
Y_pred_knn = knn.predict(X_test)
Y_pred_nb = nb.predict(X_test)
Y_pred_dectree = dectree.predict(X_test)
Y_pred_ranfor = ranfor.predict(X_test)


# In[64]:


# Evaluating using accuracy_score metric
from sklearn.metrics import accuracy_score
accuracy_logreg = accuracy_score(Y_test, Y_pred_logreg)
accuracy_knn = accuracy_score(Y_test, Y_pred_knn)
accuracy_nb = accuracy_score(Y_test, Y_pred_nb)
accuracy_dectree = accuracy_score(Y_test, Y_pred_dectree)
accuracy_ranfor = accuracy_score(Y_test, Y_pred_ranfor)


# In[65]:


# Accuracy on test set
print("Logistic Regression: " + str(accuracy_logreg * 100))
print("K Nearest neighbors: " + str(accuracy_knn * 100))
print("Naive Bayes: " + str(accuracy_nb * 100))
print("Decision tree: " + str(accuracy_dectree * 100))
print("Random Forest: " + str(accuracy_ranfor * 100))


# In[66]:


# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred_ranfor)
cm


# In[67]:


# Heatmap of Confusion matrix
sns.heatmap(pd.DataFrame(cm), annot=True)


# In[68]:


# Classification report
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred_ranfor))


# In[69]:


# Plotting the ROC curve
from sklearn import metrics
sns.set_style('darkgrid')
y_pred_ranfor = ranfor.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(Y_test,  y_pred_ranfor)
auc_val = metrics.auc(fpr, tpr)
print("AUC: ", auc_val)
plt.title('ROC curve - Random Forest')
plt.plot(fpr,tpr, 'b')
plt.plot([0, 1], [0, 1], 'r--')
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.show()


# In[14]:


#XGBoost model(Tree based)
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[18]:


#XGBoost with Bagging
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

base_model = xgb.XGBClassifier()

bagging_model = BaggingClassifier(base_model, n_estimators=10, random_state=42)

bagging_model.fit(X_train, y_train)

y_pred = bagging_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[16]:


#XGBoost model(linear model)
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
model = xgb.XGBClassifier(booster='gblinear')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[17]:


#XGBoost model(dart)
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
model = xgb.XGBClassifier(booster='dart')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[28]:


from sklearn.metrics import classification_report
print(classification_report(y_test, bagging_model.predict(X_test)))


# In[30]:


# Plotting the ROC curve
from sklearn import metrics
sns.set_style('darkgrid')
y_pred = bagging_model.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred)
auc_val = metrics.auc(fpr, tpr)
print("AUC: ", auc_val)
plt.title('ROC curve - XGBoost')
plt.plot(fpr,tpr, 'b')
plt.plot([0, 1], [0, 1], 'r--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[26]:


# Tuning the hyperparamter
from skopt import BayesSearchCV
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


Y = Y.ravel()  
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
param_space = {
    'learning_rate': (0.01, 1.0, 'log-uniform'),
    'n_estimators': (100, 1000),
    'max_depth': (3, 10),
    'subsample': (0.5, 1.0),
    'colsample_bytree': (0.5, 1.0),
    'gamma': (0, 5),
    'min_child_weight': (1, 10)
}

bayes_search = BayesSearchCV(
    estimator=XGBClassifier(),
    search_spaces=param_space,
    n_iter=25,  
    cv=5,       
    n_jobs=-1,  
    verbose=1,
    random_state=42
)

bayes_search.fit(X_train, y_train)
print("Best hyperparameters found: ", bayes_search.best_params_)
print("Best score found: ", bayes_search.best_score_)


# In[16]:


pip install scikit-optimize


# In[74]:


#AdaBOOst
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = AdaBoostClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[75]:


#Gradient boosting machine 
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.datasets import load_boston, load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

gbm_regressor = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gbm_regressor.fit(X_train, y_train)

y_pred = gbm_regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

gbm_classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gbm_classifier.fit(X_train, y_train)

y_pred = gbm_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# ### Deep Learning models 

# In[ ]:


#Normal CNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Dense, Dropout, Embedding, Input

X = np.expand_dims(X, axis=2)
model = Sequential([
    Conv1D(128, 5, activation='relu', padding='valid', strides=1),
    GlobalMaxPooling1D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)



# In[21]:


#CNN with regularization
from tensorflow.keras import regularizers

X = np.expand_dims(X, axis=2) 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = Sequential([
    Conv1D(128, 3, activation='relu', padding='valid', input_shape=(X.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Conv1D(64, 3, activation='relu', padding='valid'),
    MaxPooling1D(pool_size=2),
    GlobalMaxPooling1D(),
    Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=25, batch_size=32, validation_split=0.2)
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)


# In[13]:


#simple RNN
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.optimizers import Adam

model = Sequential([
    SimpleRNN(64, input_shape=(X_train.shape[1], X_train.shape[2]), activation='relu', return_sequences=True),
    SimpleRNN(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
optimizer = Adam(lr=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)




# In[19]:


#LSTM 
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.utils import to_categorical

X = data[features].values
Y = data['Satisfaction'].values
X = np.expand_dims(X, axis=2)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Define LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(Y.shape[1], activation='sigmoid'))  

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=20, batch_size=32, validation_data=(X_test, Y_test), verbose=2)


# In[ ]:


#LSTM 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.utils import to_categorical

# Assuming 'data' is your DataFrame containing the features and target variable
features=['Online Boarding', 'In-flight Wifi Service', 'Class', 'Type of Travel', 'In-flight Entertainment', 
          'Seat Comfort', 'Flight Distance', 'Leg Room Service', 'Customer Type', 'Ease of Online Booking', 
          'On-board Service', 'Cleanliness', 'Check-in Service', 'In-flight Service', 'Baggage Handling']

# Extracting features and target variable
X = data[features].values
Y = data['Satisfaction'].values

# Encode target labels if necessary
label_encoder = LabelEncoder()
Y = to_categorical(label_encoder.fit_transform(Y))

# Reshape input features for LSTM
X = np.expand_dims(X, axis=2)

# Splitting the dataset into the Training set and Test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Define LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(Y.shape[1], activation='sigmoid'))  

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit model
model.fit(X_train, Y_train, epochs=20, batch_size=32, validation_data=(X_test, Y_test), verbose=2)

