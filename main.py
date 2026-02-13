import pandas as pd 
import numpy as np 

df = pd.read_csv("insurance - insurance.csv")
# print(df.head(3))

# print(df.isnull().sum())

from sklearn.model_selection import train_test_split

x = df.drop(columns=['charges'])
y = df['charges']

x_train, x_test, y_train, y_test =train_test_split(x, y, test_size=0.2, random_state=87)

# print(df.shape)
# print(x_train.shape)
# print(x_test.shape)

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

ohe = OneHotEncoder(drop = 'first' , sparse_output = False )
x_train_sex_smoker_region = ohe.fit_transform(x_train[['sex' , 'smoker','region']])
x_test_sex_smoker_region = ohe.fit_transform(x_test[['sex' , 'smoker','region']])
print(x_train_sex_smoker_region.shape)

x_train_age_bmi_children = x_train.drop(columns =['smoker', 'region','sex']).values
x_test_age_bmi_children = x_test.drop(columns =['smoker', 'region','sex']).values
print(x_train_age_bmi_children.shape)

x_train_transformed = np.concatenate((x_train_age_bmi_children ,x_train_sex_smoker_region) , axis = 1)
print(x_train_transformed.shape)

from sklearn.compose import ColumnTransformer

transformers = ColumnTransformer(transformers=[
    ('tnf1',OneHotEncoder(sparse_output=False, drop='first'),['sex','smoker','region'])
],remainder='passthrough')

print(transformers.fit_transform(x_train).shape)
