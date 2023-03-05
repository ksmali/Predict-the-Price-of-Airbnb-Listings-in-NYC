
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score




# Load the dataset
df = pd.read_csv('listings.csv', low_memory = False)


print(df.head())
print(df.info())
print(df.describe())



# Data cleaning and preparation
# Remove duplicate entries
df = df.drop_duplicates()

# Remove irrelevant columns
df = df.drop(['id', 'name', 'host_id', 'host_name', 'last_review'], axis=1)

# Handle missing values
df = df.fillna({'reviews_per_month': 0})




# Convert data types
df['price'] = df['price'].astype(str).str.replace(',', '').str.replace('$', '').astype(float)




# Visualization
# Histogram of the price distribution
plt.hist(df.price, bins=50)
plt.xlabel('Price ($)')
plt.ylabel('Count')
plt.title('Price Distribution of Airbnb Listings in NYC')
plt.show()



# Scatterplot of price vs. availability
plt.scatter(df.availability_365, df.price)
plt.xlabel('Availability (Days)')
plt.ylabel('Price ($)')
plt.title('Price vs. Availability of Airbnb Listings in NYC')
plt.show()



# Calculate the average price per neighborhood group
avg_price_by_ng = df.groupby('neighbourhood_group')['price'].mean().reset_index()




# Bar chart of average price per neighborhood group
sns.barplot(x='neighbourhood_group', y='price', data=avg_price_by_ng)
plt.xlabel('Neighborhood Group')
plt.ylabel('Average Price ($)')
plt.title('Average Price of Airbnb Listings by Neighborhood Group in NYC')
plt.show()


sns.boxplot(x='neighbourhood_group', y='price', data=df)
plt.xlabel('Neighborhood Group')
plt.ylabel('Price ($)')
plt.title('Price Distribution of Airbnb Listings by Neighborhood Group in NYC')
plt.show()

# Model training and evaluation
# Split data into training and testing sets
X = df[['latitude', 'longitude', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365']]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



count_by_ng = df.groupby('neighbourhood_group')['neighbourhood'].count().reset_index()
sns.barplot(x='neighbourhood_group', y='neighbourhood', data=count_by_ng)
plt.xlabel('Neighborhood Group')
plt.ylabel('Number of Listings')
plt.title('Number of Airbnb Listings by Neighborhood Group in NYC')
plt.show()



# Decision Tree Regression
dt_reg = DecisionTreeRegressor(max_depth=5)
dt_reg.fit(X_train, y_train)
y_pred_dt = dt_reg.predict(X_test)
mse_dt = mean_squared_error(y_test, y_pred_dt)
print('Mean squared error of Decision Tree Regression: ', mse_dt)



# Random Forest Regression
rf_reg = RandomForestRegressor(n_estimators=100, max_depth=5)
rf_reg.fit(X_train, y_train)
y_pred_rf = rf_reg.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
print('Mean squared error of Random Forest Regression: ', mse_rf)

# Visualize the results
plt.scatter(y_test, y_pred_dt, label='Decision Tree')
plt.scatter(y_test, y_pred_rf, label='Random Forest')
plt.plot([0, max(y_test)], [0, max(y_test)], linestyle='--', color='black')
plt.xlabel('Actual Price ($)')
plt.ylabel('Predicted Price ($)')
plt.title('Actual vs. Predicted Prices of Airbnb Listings in NYC')
plt.legend()
plt.show()



# Calculate the average price and number of listings by neighborhood group
avg_price_by_ng = df.groupby('neighbourhood_group')['price'].mean().reset_index()
count_by_ng = df.groupby('neighbourhood_group')['neighbourhood'].count().reset_index()

# Create subplots
fig = make_subplots(rows=1, cols=2, subplot_titles=('Average Price by Neighborhood Group', 'Number of Listings by Neighborhood Group'))

# Add traces to subplots
fig.add_trace(go.Bar(x=avg_price_by_ng['neighbourhood_group'], y=avg_price_by_ng['price'], name='Price'), row=1, col=1)
fig.add_trace(go.Bar(x=count_by_ng['neighbourhood_group'], y=count_by_ng['neighbourhood'], name='Listings'), row=1, col=2)

# Update layout
fig.update_layout(title='Airbnb Listings in NYC by Neighborhood Group', barmode='group')

# Show dashboard
fig.show()




# Calculate the average price per neighborhood group and room type
avg_price_by_ng_rt = df.groupby(['neighbourhood_group', 'room_type'])['price'].mean().reset_index()

# Calculate the average availability per neighborhood group and room type
avg_avail_by_ng_rt = df.groupby(['neighbourhood_group', 'room_type'])['availability_365'].mean().reset_index()





# Visualization
# Histogram of the price distribution
plt.hist(df.price, bins=50)
plt.xlabel('Price ($)')
plt.ylabel('Count')
plt.title('Price Distribution of Airbnb Listings in NYC')
plt.show()

# Scatterplot of price vs. availability
plt.scatter(df.availability_365, df.price)
plt.xlabel('Availability (Days)')
plt.ylabel('Price ($)')
plt.title('Price vs. Availability of Airbnb Listings in NYC')
plt.show()


# Bar chart of average price per neighborhood group and room type
sns.barplot(x='neighbourhood_group', y='price', hue='room_type', data=avg_price_by_ng_rt)
plt.xlabel('Neighborhood Group')
plt.ylabel('Average Price ($)')
plt.title('Average Price of Airbnb Listings by Neighborhood Group and Room Type in NYC')
plt.show()

# Bar chart of average availability per neighborhood group and room type
sns.barplot(x='neighbourhood_group', y='availability_365', hue='room_type', data=avg_avail_by_ng_rt)
plt.xlabel('Neighborhood Group')
plt.ylabel('Average Availability (Days)')
plt.title('Average Availability of Airbnb Listings by Neighborhood Group and Room Type in NYC')
plt.show()


# Model training and evaluation
# Split data into training and testing sets
X = df[['latitude', 'longitude', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365']]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree Regression
dt_reg = DecisionTreeRegressor(max_depth=5)
dt_reg.fit(X_train, y_train)
y_pred_dt = dt_reg.predict(X_test)
mse_dt = mean_squared_error(y_test, y_pred_dt)
print('Mean squared error of Decision Tree Regression: ', mse_dt)



# Visualize the results
plt.scatter(y_test, y_pred_dt, label='Decision Tree')
plt.scatter(y_test, y_pred_rf, label='Random')


mse_dt = mean_squared_error(y_test, y_pred_dt)
accuracy_dt = round(dt_reg.score(X_test, y_test), 3)
f1_dt = f1_score(y_test, y_pred_dt.round(), average='micro')

print('Mean squared error of Decision Tree Regression:', mse_dt)
print('Accuracy of Decision Tree Regression:', accuracy_dt)
print('F1-score of Decision Tree Regression:', f1_dt)

# Random Forest Regression
rf_reg = RandomForestRegressor(n_estimators=500, max_depth=5)
rf_reg.fit(X_train, y_train)
y_pred_rf = rf_reg.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)


# Random Forest Regression

mse_rf = mean_squared_error(y_test, y_pred_rf)
accuracy_rf = round(rf_reg.score(X_test, y_test), 3)
f1_rf = f1_score(y_test, y_pred_rf.round(), average='micro')

print('Mean squared error of Random Forest Regression:', mse_rf)
print('Accuracy of Random Forest Regression:', accuracy_rf)
print('F1-score of Random Forest Regression:', f1_rf)

