# !pip install pandas

from urllib.request import urlretrieve # downloads data using url

medical_charges_url = 'https://raw.githubusercontent.com/JovianML/opendatasets/master/data/medical-charges.csv'

urlretrieve(medical_charges_url, 'medical.csv')

import pandas as pd
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

medical_df = pd.read_csv('medical.csv')
medical_df

medical_df.info()

medical_df.describe()

# !pip install plotly matplotlib seaborn

sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'

medical_df.age.describe()

fig = px.histogram(
    medical_df,
    x='age',
    marginal='box',
    nbins=45,
    title='Age Distribution'
)
fig.update_layout(bargap=0.1)
fig.show()

fig = px.histogram(medical_df,
                   x='bmi',
                   marginal='box',
                   color_discrete_sequence=['red'],
                   title='Distribution of BMI (Body Mass Index)')
fig.update_layout(bargap=0.1)
fig.show()

fig = px.histogram(medical_df,
                   x='charges',
                   marginal='box',
                   color='smoker',
                   color_discrete_sequence=['green', 'grey'],
                   title='Annual Medical Charges')
fig.update_layout(bargap=0.1)
fig.show()

medical_df.smoker.value_counts()

px.histogram(medical_df, x='smoker', color='sex', title='Smoker')

fig = px.scatter(medical_df,
                 x='age',
                 y='charges',
                 color='smoker',
                 opacity=0.8,
                 hover_data=['sex'],
                 title='Age vs. Charges')
fig.update_traces(marker_size=5)
fig.show()

fig = px.scatter(medical_df,
                 x='bmi',
                 y='charges',
                 color='smoker',
                 opacity=0.8,
                 hover_data=['sex'],
                 title='BMI vs. Charges')
fig.update_traces(marker_size=5)
fig.show()

medical_df.charges.corr(medical_df.age) # correlation b/w charges and age

medical_df.charges.corr(medical_df.bmi)

smoker_values = {'no': 0, 'yes': 1} # categorical data converted into 0s and 1s
smoker_numeric = medical_df.smoker.map(smoker_values) # new column with 0s and 1s
medical_df.charges.corr(smoker_numeric) # correlation b/w charges and smoker

#medical_df['age','bmi','children','charges'].corr()

# !pip install scikit-learn

from sklearn.linear_model import LinearRegression

model = LinearRegression()

help(model.fit)

smoker_numeric = medical_df.smoker.map(smoker_values)
medical_df['smoker_num'] = smoker_numeric

sex_values = {'male': 0, 'female': 1}
sex_numeric = medical_df.sex.map(sex_values)
medical_df['sex_num'] = sex_numeric

inputs, target = medical_df[['age', 'bmi', 'children', 'smoker_num', 'sex_num']], medical_df.charges

model.fit(inputs, target)

predictions = model.predict(inputs)

predictions

import math
import sklearn.metrics as metrics

def rmse(targets, predictions):
    return math.sqrt(metrics.mean_squared_error(targets, predictions))

loss = rmse(target, predictions)
loss

# one-hot encoding

sns.barplot(data=medical_df, x='region', y='charges');

from sklearn import preprocessing
enc = preprocessing.OneHotEncoder()
enc.fit(medical_df[['region']])
enc.categories_

one_hot = enc.transform(medical_df[['region']]).toarray()
one_hot

medical_df[['northeast', 'northwest', 'southeast', 'southwest']] = one_hot
medical_df

input_cols = ['age', 'bmi', 'children', 'northeast', 'northwest', 'southeast', 'southwest', 'sex_num', 'smoker_num']
inputs_1, target = medical_df[input_cols], medical_df.charges

model.fit(inputs_1, target)

predictions_1 = model.predict(inputs_1)

loss_1 = rmse(target, predictions_1)
loss_1 # very less reduction in loss

model.coef_

model.intercept_

import numpy as np

weights_df = pd.DataFrame({
    'feature': np.append(input_cols, 1),
    'weight': np.append(model.coef_, model.intercept_)
})
weights_df

from sklearn.preprocessing import StandardScaler

numeric_cols = ['age', 'bmi', 'children']
scaler = StandardScaler()
scaler.fit(medical_df[numeric_cols])

scaler.mean_

scaler.var_

scaled_inputs = scaler.transform(medical_df[numeric_cols])
scaled_inputs

cat_cols = ['smoker_num', 'sex_num', 'northeast', 'northwest', 'southeast', 'southwest']
categorical_data = medical_df[cat_cols].values

inputs_2 = np.concatenate((scaled_inputs, categorical_data), axis=1)
target = medical_df.charges

model.fit(inputs_2, target)

predictions_2 = model.predict(inputs_2)

loss_2 = rmse(target, predictions_2)
loss_2

weights_df = pd.DataFrame({
    'feature': np.append(numeric_cols + cat_cols, 1),
    'weight': np.append(model.coef_, model.intercept_)
})
weights_df.sort_values('weight', ascending=False)

from sklearn.model_selection import train_test_split

inputs_train, inputs_test, target_train, target_test = train_test_split(inputs_2, target, test_size=0.1)

model.fit(inputs_train, target_train)

predictions_test = model.predict(inputs_test)

loss_3_test = rmse(target_test, predictions_test)
loss_3_test

predictions_train = model.predict(inputs_train)

loss_3_train = rmse(target_train, predictions_train)
loss_3_train


