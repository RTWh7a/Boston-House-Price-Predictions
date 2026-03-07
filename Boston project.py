#Import library for project
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport

#data loading 
df=pd.read_csv(r"C:\Users\rafaa\Downloads\BostonHousing.csv")
print(df.head())

#Data preprocess
print(df.isna().sum())
print(df.info())
print(df.describe())

#Data cleaning
plt.boxplot(df,labels=df.columns)
plt.show()

#IQR
data=df.copy()
Q1=df.quantile(0.25)
Q3=df.quantile(0.75)
IQR=Q3-Q1
outliers=data[((data<(Q1-1.5*IQR))|(data>(Q3+1.5*IQR))).any(axis=1)]
clean_data=data[~((data<(Q1-1.5*IQR))|(data>(Q3+1.5*IQR))).any(axis=1)]
plt.boxplot(clean_data,labels=clean_data.columns)
plt.show()

#EDA plots dashboard
#box plot MEDV
plt.figure(figsize=(10, 10))
sns.scatterplot(x=clean_data['rm'], y=clean_data['medv'], color='blue',data=clean_data, alpha=0.5)
plt.xlabel('Average Number of Rooms per Dwelling (RM)')
plt.ylabel('Median Value of Owner-Occupied Homes (MEDV)')
plt.title('Scatter Plot of Median Value of Owner-Occupied Homes (MEDV) vs Average Number of Rooms')
plt.show()
#count plot of CHAS
plt.figure(figsize=(10, 6))
sns.countplot(x=clean_data['chas'], palette='viridis', order=clean_data['chas'].value_counts().index)
plt.xlabel('Charles River Dummy Variable (CHAS)')
plt.ylabel('Count')
plt.title('Count of Charles River Dummy Variable (CHAS)')
plt.show()

#correlation heatmap
corr=clean_data.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5, linecolor='white', cbar_kws={'shrink': 0.5}, annot_kws={'size': 10})
plt.title('Correlation Heatmap of Boston Housing Dataset')
plt.show()

#dashboard of data 
fig,axis=plt.subplots(2,3,figsize=(30,30))
sns.histplot(clean_data['medv'], bins=30, kde=True, color='blue', ax=axis[0,0])
axis[0,0].set_title('Distribution of MEDV')
sns.scatterplot(x=clean_data['rm'], y=clean_data['medv'], color='red', alpha=0.5, ax=axis[0,1])
axis[0,1].set_title('MEDV vs RM')
sns.countplot(x=data['chas'], palette='viridis', order=data['chas'].value_counts().index, ax=axis[0,2])
axis[0,2].set_title('Count of CHAS')
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', linewidths=0.5, linecolor='white', cbar_kws={'shrink': 0.5}, annot_kws={'size': 8}, ax=axis[1,0])
axis[1,0].set_title('Correlation Heatmap')
sns.countplot(x=clean_data['rad'], palette='viridis', order=clean_data['rad'].value_counts().index, ax=axis[1,1])
axis[1,1].set_title('Count of RAD')
sns.boxplot(x=clean_data['medv'], color='green', ax=axis[1,2])
axis[1,2].set_title('Box Plot of MEDV')
plt.tight_layout()
plt.show()

#Build model
from sklearn.model_selection import train_test_split
X =clean_data[['rm']].iloc[:,:].values
y=clean_data['medv']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#shape of data
print(X.shape)
print(x_train.shape)
print(x_test.shape)
print()
print(y.shape)
print(y_train.shape)
print(y_test.shape)
#Scale data        
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
    
#Model Build
pipelines = make_pipeline(StandardScaler(), PCA(),LinearRegression())
pipelines.fit(x_train, y_train)

#Evaluate model
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error

y_pred=pipelines.predict(x_test)

print(f'R2_score: {r2_score(y_test, y_pred)}')
print(f'MSE: {mean_squared_error(y_test, y_pred)}')
print(f'MAE: {mean_absolute_error(y_test, y_pred)}')
print(f'Predicted values: {y_pred}')
print(f'Actual values: {y_test.values}')
print(f'Coefficients: {pipelines.named_steps["linearregression"].coef_}')
print(f'Intercept: {pipelines.named_steps["linearregression"].intercept_}')
print(f'Model Accuracy: {pipelines.score(x_test, y_test)*100:.2f}%')

#linear regression plot
plt.figure(figsize=(10, 6))
plt.scatter(x_train,y_train, color='blue', label='Training Data', alpha=0.5)
plt.scatter(x_test, y_test, color='green', label='Testing Data', alpha=0.5)
plt.plot(x_test, y_pred, color='red', label='Predicted Regression Line')
plt.xlabel('Average Number of Rooms per Dwelling (RM)')
plt.ylabel('Median Value of Owner-Occupied Homes (MEDV)')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.show()

#PCA plot
plt.figure(figsize=(10, 6))
plt.scatter(x_train[:, 0], y_train, color='blue', label='Training Data', alpha=0.5)
plt.scatter(x_test[:, 0], y_test, color='green', label='Testing Data', alpha=0.5)
plt.xlabel('Average Number of Rooms per Dwelling (RM)')
plt.legend(loc='best')
plt.title('PCA Plot of Training and Testing Data')
plt.grid(True)
plt.tight_layout()
plt.show()

#Learning curve & Validation curve
from sklearn.model_selection import learning_curve, validation_curve

#plot each model learning curve
models={
    'Linear Regression': LinearRegression(),
    'PCA':PCA()
}
for model_name,model in models.items():
    train_sizes, train_scores, test_scores = learning_curve(model, x_train, y_train, cv=5, n_jobs=-1, 
                                                            train_sizes=np.linspace(0.1, 1.0, 10), random_state=42)
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    train_scores_std=np.std(train_scores, axis=1)
    test_scores_std=np.std(test_scores, axis=1)

    plt.figure()
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation score')
    plt.fill_between(train_sizes, train_scores_mean-train_scores_std,
                    train_scores_mean+train_scores_std,alpha=0.1,color='r')
    plt.fill_between(train_sizes, test_scores_mean-test_scores_std,
                    test_scores_mean+test_scores_std,alpha=0.1,color='g')
    plt.title(f'Learning Curve for {model_name}')
    plt.xlabel('Training Size')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.grid()
    plt.show()

#dendogram plot compind with heat_mape
plt.figure(figsize=(10, 6))
from scipy.cluster.hierarchy import dendrogram, linkage
linked = linkage(clean_data, method='ward')

dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Distance')
plt.ylabel('Sample Index')
plt.grid(True)
#heat_map compind with dendogram
plt.figure(figsize=(10, 6))
sns.heatmap(clean_data.corr(), annot=True, cmap='coolwarm',
 linewidths=0.5, linecolor='white', cbar_kws={'shrink': 0.5}, annot_kws={'size': 8})
plt.title('Correlation Heatmap')
plt.grid(True)
plt.show()

#Compare the Price of RM and its preidect
sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(x_train)
y_std = sc_y.fit_transform(y_train.values.reshape(-1, 1))
lr = LinearRegression()
lr.fit(X_std, y_std)

num_rooms_std = sc_x.transform(np.array([[5.0]]))
price_std = lr.predict(num_rooms_std)
original_price=sc_y.inverse_transform(price_std.reshape(-1,1))
print(f"Price in $1000s: {original_price[0][0]}")
