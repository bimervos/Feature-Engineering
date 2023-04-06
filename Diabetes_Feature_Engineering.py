

##################    About Dataset    ##################

# This dataset is a part of a large dataset held at the National Institute of Diabetes and Digestive and Kidney Diseases, and is used for a diabetes study conducted on Pima  Indian women aged 21 and above living in Phoenix, the fifth-largest city in the state of Arizona in the United States. The target variable is specified as "outcome", where 1 indicates a positive result of the diabetes test and 0 indicates a negative result.
#
# A machine learning model is requested to be developed that can predict whether individuals have diabetes or not when their features are given. Before developing the model, performing the necessary data analysis and feature engineering steps are expected.
#
# There are 9 variables in the dataset.
# Pregnancies: Number of pregnancies.
# Glucose: 2-hour plasma glucose concentration during an oral glucose tolerance test.
# Blood Pressure: Blood Pressure (Diastolic) (mm Hg).
# SkinThickness : Thickness of skin
# Insulin: 2-Hour Serum Insulin (mu U/ml)
# DiabetesPedigreeFunction: Function (2-hour plasma glucose concentration during an oral glucose tolerance test).
# BMI: Body Mass Index (weight in kg/(height in meters)^2)
# Age: Persons age(in years)
# Outcome: Having diabetes status (1: diabetes, 0: no diabetes)


#Import the libraries :

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
#!pip install missingno
import missingno as msno
from datetime import date
#!pip install ydata-profiling
#from ydata-profiling import ProfileReport
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

#Some configurations

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

#Load the dataset :

df= pd.read_csv('Case_Study/diabetes.csv')
df.head()

##################    Exploratory Data Analysis    ##################

#In exploratory data analysis, the type of the variables (e.g. integer or decimal) is first printed, the sum of the missing values (either null or na) is calculated, and it is found that no such value exists.
# In addition, some statistical measures are calculated to understand their variation. Some of these statistical measures are the mean of each column, the median, the standard deviation, the percentages, etc.

df.shape

df.info()

#Is there any missing value in the dataset?
df.isnull().any()

# Concise statistical description of numeric features
df.describe().T

#Are there any duplicate values?
df.duplicated().any()
# There are no missing or duplicated observations in the dataset in general. The variable types also appear to be correct.

# profile = ProfileReport(df, title="Diabetes Classification")
# profile


#Now let's apply the function that will separate the numerical and categorical variables for us:

def grab_col_names (dataframe, cat_trh=10, car_thr=20):
    cat_cols= [col for  col in dataframe.columns if dataframe[col].dtype =='O']
    num_but_cat=[col for col in dataframe.columns if dataframe[col].nunique() < cat_trh and dataframe[col].dtype != 'O']
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_thr and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols= [col for col in  dataframe.columns if dataframe[col].dtype != 'O']
    num_cols = [col for col in num_cols if col not in num_but_cat]

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)


##################    Categorical Variable Analysis    ##################

def cat_summary( dataframe, col_name, plot = False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                       'Ratio':100 * dataframe[col_name].value_counts()/len(dataframe)}))
    print('#####################',col_name,'############################')

    if plot:
        sns.countplot(x = dataframe[col_name], data = dataframe, edgecolor='black', color='#D6B2B1', saturation= 0.5)
        plt.show(block = True)

for col in cat_cols:
    cat_summary(df, col, plot = True)
for col in num_cols:
    cat_summary(df, col, plot = True)


##################    Target Veriable Analysis    ##################

import warnings
warnings.filterwarnings("ignore")

columns = df.columns
columns = list(columns)
columns.pop()
print("Column names except for the target column are :", columns)

# Graphs to be plotted with these colors
colours = ['b', 'c', 'g', 'k', 'm', 'r', 'y', 'b']
print()
print('Colors for the graphs are :', colours)
sns.set(rc={'figure.figsize': (15, 17)})
colors_list = ['#78C850', '#F08030']
j = 1
sns.set_style(style='white')

for i in (columns):
    plt.subplot(4, 2, j)
    sns.violinplot(x="Outcome", y=i, data=df, kind="violin", split=True, height=4, aspect=.7, palette=colors_list)
    sns.swarmplot(x='Outcome', y=i, data=df, color="k", alpha=0.8)
    j = j + 1
plt.show(block=True)



# Let's look at the means of the numerical variables according to the target variable.
for col in num_cols:
    print(df.groupby('Outcome').agg({col: 'mean'}))

for col in num_cols:
    print(df.groupby(col)['Outcome'].mean())

f, axs = plt.subplots(1, len(num_cols), figsize=(19, 6), constrained_layout=True)
for i, col in enumerate(num_cols):
    sns.distplot(df[col], bins=20, ax=axs[i], color='#D6B2B1')
plt.show(block=True)


##################    Outlier Analysis    ##################

#Let's determine the necessary lower and upper limits to conduct outlier analysis

def outlier_thresholds(dataframe, col_name, q1=0.10, q3=0.90):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


#Let's write a function that returns True if there is a value outside of the lower and upper limits that we have set for the function's variables.

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

check_outlier(df, cat_cols)
check_outlier(df, num_cols)

#We can see the outliers on boxplot:

fig, axes = plt.subplots(2, 4, figsize=(24, 10))
for i, col in enumerate(num_cols):
    sns.boxplot(x=df[col], ax=axes[i//4, i%4], color='#D6B2B1' )
    axes[i//4, i%4].set_title(col)
plt.show(block=True)


#Let's suppress the outliers according to the lower and upper limits we have created.

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in df[num_cols]:
    replace_with_thresholds(df, col)

#When we check again, we will see that there are no outliers:

check_outlier(df, num_cols)

# There may also be values that are not outliers on their own but may become outliers when evaluated together.
# For example, being 20 years old is not an outlier value, but being pregnant for the 12th time at the age of 20 is a significant event that may be considered an outlier.
# That is why we should also evaluate such situations with Local Outlier Factor:
#
# LOF calculates a local density value for each data point. This value is determined based on the density of the data point's neighbors. The density of a point is determined based on the densities of its surrounding neighbor points.
# Then, LOF compares each point's local density value with the densities of its neighboring points. Outliers are defined as points with much lower density than their surrounding points.LOF is an effective outlier detection method, especially for complex and high-dimensional datasets.

clf= LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df)
df_scores= clf.negative_outlier_factor_ #scores are ready.

fig, ax = plt.subplots(figsize=(10, 10))
scores= pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0,20], style='.-', ax=ax)
plt.show(block=True)

#As seen in the graph, the first and largest break occurs up to the 4th value. We will select this value as the threshold point.

th= np.sort(df_scores)[4]

#And here is the outliers: (Multiple)

df[df_scores < th]

#Since there are not too many outliers, we will delete them:

df.shape
df = df[~(df_scores < th)]

# for col in df[num_cols]:
#     sns.boxplot(df[col])
#     plt.show(block=True)


##################    Correlation Matrix For Numeric Features    ##################

plt.subplots(figsize=(10,7))
sns.heatmap(df.corr(), vmax=0.9, annot=True, square=True,  cmap='copper_r')
plt.show(block=True)

plt.figure(dpi = 120,figsize= (5,4))
mask = np.triu(np.ones_like(df.corr(),dtype = bool))
sns.heatmap(df.corr(),mask = mask, fmt = ".2f",annot=True,lw=1,cmap = 'copper_r')
plt.yticks(rotation = 0)
plt.xticks(rotation = 90)
plt.title('Correlation Heatmap')
plt.show(block=True)

#When we examine the graph, we observe that the highest correlation is between the variables of pregnancy-age , glucose-outcome and insulin-skin thickness.

# sns.pairplot(df, hue='Outcome', palette='copper_r')
# plt.suptitle('Feature Relationship', y=1.05, weight='bold', fontsize=5)
# plt.show(block=True)
#
# fig, axes = plt.subplots(nrows=1, ncols=len(num_cols))
# for i, col in enumerate(num_cols):
#     sns.regplot(x=df[col], y=df['Outcome'], ax=axes[i])
#     axes[i].set_title(col)
# plt.show(block=True)

##################    Mıssıng Values     ##################

df.isnull().any()

#We had analyzed that there were no missing values in the dataset. However, we can observe that there are meaningless values in the dataset.
#For example: Variables such as Glucose, Insulin, Skin Thickness and Blood Pressure cannot be 0.
#Therefore, we might need to count them as NAN. Let's check descripteves of the variables again.

df.describe().T

df.loc[ df['Insulin'] == 0, 'Insulin'] = np.nan
df.loc[ df['SkinThickness'] == 0, 'SkinThickness'] = np.nan
df.loc[ df['Glucose'] == 0, 'Glucose'] = np.nan
df.loc[ df['BMI'] == 0, 'BMI'] = np.nan

#Now let's fill in these missing values by estimating them based on similar values with k-NN.

dff= pd.get_dummies(df, drop_first=True)  #Categorical variables were expressed as numeric variables.

#Standardization of variables.
scaler= MinMaxScaler()
dff= pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)

#Filling in missing values  with KNN.
from sklearn.impute import KNNImputer
imputer= KNNImputer(n_neighbors=5)
dff= pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)

dff= pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns) #We made the data readable by reversing the standardization.

#Let's observe the missing values and the new filled values in Insulin veriable:

df['Insulin_Imputed_KNN'] = dff[['Insulin']]
df.loc[df['Insulin'].isnull()]

############################# Generating New Features ####################################

df = dff
df.head()

#When we examine the graph, we observe that the highest correlation is between the variables of pregnancy-age , glucose-outcome and insulin-skin thickness.

df['Pregnancies'].value_counts()
df.loc[ (df['Pregnancies'] == 0 ) , 'PREGNANT_COUNT_NEW' ] = 'no_pregnant'
df.loc[ (df['Pregnancies'] == 1 ) , 'PREGNANT_COUNT_NEW' ] = 'first_pregnant'
df.loc[ ((df['Pregnancies'] > 1 ) & (dff['Pregnancies'] < 10 )), 'PREGNANT_COUNT_NEW' ] = 'many_pregnant'
df.loc[ (df['Pregnancies'] >= 10 ), 'PREGNANT_COUNT_NEW' ] = 'much_pregnant'

df["AGE*DIABETESPEDIGREE_NEW"] = df["Age"] * df["DiabetesPedigreeFunction"]

df["AGE*BMI_NEW"] = df["Age"] * df["BMI"]

df['Glucose'].unique()
df["GLUCOSE*INSULIN_NEW"] = df["Glucose"] * df["Insulin"]
df['GLUCOSE*INSULIN_CUT_NEW']= pd.cut(df['GLUCOSE*INSULIN_NEW'], bins = [1000, 9000,  16000, 25000, 105000], labels=['level_a', 'level_b', 'level_c', 'level_d',])
df['GLUCOSE_CUT_NEW']= pd.cut(df['Glucose'], bins = [40, 140, 200], labels=['low_glucose', 'high_glukose'])

df["SKIN*INSULIN_NEW"] = df["SkinThickness"] * df["Insulin"]
df['SKIN*INSULIN_CUT_NEW']= pd.cut(df['SKIN*INSULIN_NEW'], bins = [190, 4000, 42000], labels=['level_a', 'level_b'])
df["SKIN*INSULIN_NEW"].describe()

df['BloodPressure'].describe()
df['BLOOD_CUT_NEW']= pd.cut(df['BloodPressure'], bins = [0, 70, 125], labels=['low_blood_pressure', 'high_blood_pressure'])

df.groupby('Outcome').agg({'SkinThickness':'mean'})
df['SKIN_THICK_NEW']= pd.cut(df['SkinThickness'], bins = [7, 30, 99], labels=['low_skin_thickness', 'high_skin_thickness'])

df['BMI'].describe()
df['BMI_CUT_NEW']= pd.cut(df['BMI'], bins = [15, 25, 30, 35, 40, 70], labels=['normal_weight','overweight', 'class_1_obesity', 'class_2_obesity', 'class_3_obesity'])

df.loc[(df["BMI"] < 18.5) & ((df["Age"] >= 21) & (df["Age"] < 50)), "AGE_BMI_NOM_NEW"] = "underweightmature"
df.loc[(df["BMI"] < 18.5) & (df["Age"] >= 50), "AGE_BMI_NOM_NEW"] = "underweightsenior"
df.loc[((df["BMI"] >= 18.5) & (df["BMI"] < 25)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "AGE_BMI_NOM_NEW"] = "healthymature"
df.loc[((df["BMI"] >= 18.5) & (df["BMI"] < 25)) & (df["Age"] >= 50), "AGE_BMI_NOM_NEW"] = "healthysenior"
df.loc[((df["BMI"] >= 25) & (df["BMI"] < 30)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "AGE_BMI_NOM_NEW"] = "overweightmature"
df.loc[((df["BMI"] >= 25) & (df["BMI"] < 30)) & (df["Age"] >= 50), "AGE_BMI_NOM_NEW"] = "overweightsenior"
df.loc[(df["BMI"] > 18.5) & ((df["Age"] >= 21) & (df["Age"] < 50)), "AGE_BMI_NOM_NEW"] = "obesemature"
df.loc[(df["BMI"] > 18.5) & (df["Age"] >= 50), "AGE_BMI_NOM_NEW"] = "obesesenior"

df.loc[(df["Glucose"] < 70) & ((df["Age"] >= 21) & (df["Age"] < 50)), "AGE_GLUCOSE_NOM_NEW"] = "lowmature"
df.loc[(df["Glucose"] < 70) & (df["Age"] >= 50), "AGE_GLUCOSE_NOM_NEW"] = "lowsenior"
df.loc[((df["Glucose"] >= 70) & (df["Glucose"] < 100)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "AGE_GLUCOSE_NOM_NEW"] = "normalmature"
df.loc[((df["Glucose"] >= 70) & (df["Glucose"] < 100)) & (df["Age"] >= 50), "AGE_GLUCOSE_NOM_NEW"] = "normalsenior"
df.loc[((df["Glucose"] >= 100) & (df["Glucose"] <= 125)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "AGE_GLUCOSE_NOM_NEW"] = "hiddenmature"
df.loc[((df["Glucose"] >= 100) & (df["Glucose"] <= 125)) & (df["Age"] >= 50), "AGE_GLUCOSE_NOM_NEW"] = "hiddensenior"
df.loc[(df["Glucose"] > 125) & ((df["Age"] >= 21) & (df["Age"] < 50)), "AGE_GLUCOSE_NOM_NEW"] = "highmature"
df.loc[(df["Glucose"] > 125) & (df["Age"] >= 50), "AGE_GLUCOSE_NOM_NEW"] = "highsenior"

df['AGE_PREGNACIES_NEW'] = df['Pregnancies']*df['Age']
df['AGE_PREGNACIES_NEW'].describe()
df['AGE_PREGNACIES_NEW_CAT']= pd.cut(df['AGE_PREGNACIES_NEW'], bins = [-1, 26, 83,249, 801], labels=['A', 'B', 'C', 'D'])

df['AGE_CAT_NEW']= pd.cut(df['Age'], bins = [20, 30 ,40, 81], labels=['young', 'middle', 'senior'])
df['Age'].describe()

df.head()
df.isnull().sum()


############################# Encoding ####################################

#Label Encoding
df.info()
df['GLUCOSE_CUT_NEW'].unique()

def label_encoder( dataframe, veriable):
    labelencoder=  LabelEncoder()
    dataframe[veriable]=labelencoder.fit_transform(dataframe[veriable])
    return dataframe

binary_cols= [col for col in df.columns if df[col].dtype != ['float', 'int'] and df[col].nunique == 2]
#in this  dataset, binary _cols list is empty

for col in binary_cols:
    df= label_encoder(df, col)

#One-Hot Encoding

def one_hot_encoder(dataframe, categorical_cols, drop_first=True ):
    dataframe= pd.get_dummies(data=dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

ohe_cols = [col for col in  df.columns if  10 > df[col].nunique() >= 2 and col != 'Outcome']

df= one_hot_encoder(df, ohe_cols)


############################# Feature Scaling ####################################

rs= RobustScaler()
df[num_cols]= rs.fit_transform(df[num_cols])

############################# Modellng ####################################

y = df["Outcome"]
X = df.drop(["Outcome"], axis = 1)

from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
rf_model = RandomForestClassifier().fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)


def plot_importance(model, features, num=len(X)):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize = (10, 10))
    sns.set(font_scale = 1)
    sns.barplot(x = "Value", y = "Feature", data = feature_imp.sort_values(by = "Value", ascending = False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)

plot_importance(rf_model, X_train)


