import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error


# Read data and filter it
df = pd.read_csv('ratings.csv')
df = df.groupby(['userId']).filter(lambda x: len(x) >= 5)
df = df.groupby(['movieId']).filter(lambda x: len(x) >= 5)

# Create the following pivot table --> [3650 filtered movies x 610 filtered users]
pivot = pd.pivot_table(df, columns="movieId", index="userId", values="rating")
pearson_corr = pivot.corr("pearson")
print(pearson_corr)

k = int(input("Please enter the k neighbors: "))
tr_size = float(input("Please enter the train size (0.1 - 0.9): "))

# Define test set and training set
X = df.drop("rating", axis=1)
y = df["rating"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=100, train_size=tr_size)

# Average K Neighbors prediction
kneigh_reg = KNeighborsRegressor(n_neighbors=k, weights="uniform")
kneigh_reg.fit(X_train, y_train)
y_prediction = kneigh_reg.predict(X_test)

mae = mean_absolute_error(y_test, y_prediction)
print("MAE using average: " + str(mae))

# K Neighbors prediction with correlation metric
kneigh_reg = KNeighborsRegressor(n_neighbors=k, metric="cosine", weights="distance")
kneigh_reg.fit(X_train, y_train)
y_prediction = kneigh_reg.predict(X_test)

mae = mean_absolute_error(y_test, y_prediction)
print("MAE using cosine distance weight: " + str(mae))

# K Neighbors prediction with nan_euclidean metric
kneigh_reg = KNeighborsRegressor(n_neighbors=k, metric="nan_euclidean", weights="distance")
kneigh_reg.fit(X_train, y_train)
y_prediction = kneigh_reg.predict(X_test)

mae = mean_absolute_error(y_test, y_prediction)
print("MAE using nan_euclidean weight: " + str(mae))
