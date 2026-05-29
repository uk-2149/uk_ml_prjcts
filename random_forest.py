from google.colab import drive
drive.mount('/content/drive')

import pandas as pd

spotify_data = pd.read_csv("/content/drive/My Drive/Spotify_data.csv")

spotify_data.head()

spotify_data.drop(columns=['Unnamed: 0'], inplace=True)

spotify_data.info()

import matplotlib.pyplot as plt
import seaborn as sns

features = ['Energy', 'Valence', 'Danceability', 'Loudness', 'Acousticness']

for feature in features:
  plt.figure(figsize=(8,5))
  sns.scatterplot(data=spotify_data, x=feature, y='Popularity')
  plt.title(f'Popularity vs {feature}')
  plt.show()

numeric_columns = spotify_data.select_dtypes(include=['float64','int64']).columns
numeric_data = spotify_data[numeric_columns]

correlation_matrix = numeric_data.corr()

plt.figure(figsize=(12,8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

for feature in features:
  plt.figure(figsize=(8,5))
  sns.histplot(data=spotify_data, x=feature, kde=True)
  plt.title(f'Distribution of {feature}')
  plt.show()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

features = ['Energy', 'Valence', 'Danceability', 'Loudness', 'Acousticness', 'Tempo', 'Speechiness', 'Liveness']
X = spotify_data[features]
y = spotify_data['Popularity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search_rf = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, refit=True, verbose=2, cv=5)

grid_search_rf.fit(X_train_scaled, y_train)

best_params_rf = grid_search_rf.best_params_

best_rf_model = grid_search_rf.best_estimator_

y_pred_rf = best_rf_model.predict(X_test_scaled)

plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred_rf, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", lw=2)
plt.xlabel('Actual Popularity')
plt.ylabel('Predicted Popularity')
plt.title('Actual Popularity vs Predicted Popularity (Best Random Forest Model)')
plt.show()
