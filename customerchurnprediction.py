import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
import mlflow

mlflow.autolog()

df = pd.read_csv("Data/telecom_churn.csv")


"""## EDA and Data Cleaning"""
df['Churn'].value_counts()
df.isnull().sum()

"""### Standarization"""


# Split the data into features (X) and target variable (y)
X = df.drop('Churn', axis=1)
y = df['Churn']

# Perform feature scaling using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

"""### Classification"""



# Split the scaled data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Create an instance of the logistic regression model
model = LogisticRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Predict the churn for the test data
y_pred = model.predict(X_test)

# Save the trained model to Google Drive
joblib.dump(model, './model/prediction1.joblib')

# Generate the classification report
report = classification_report(y_test, y_pred)
print(report)

"""So, here we can see that the recall of the 1 is very less, since the dataset is imbalance.

### Handling Imbalance dataset
"""


# Split the scaled data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Handling class imbalance using RandomOverSampler
oversampler = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

# Check the balance of the resampled data
class_counts = y_train_resampled.value_counts()
print(class_counts)

"""### Classification Model"""

# Create an instance of the logistic regression model
model1 = LogisticRegression()

# Fit the model on the training data
model1.fit(X_train_resampled, y_train_resampled)

# Predict the churn for the test data
y_pred = model1.predict(X_test)

# Save the trained model to Google Drive
joblib.dump(model1, './model/prediction2.joblib')

#classification report
report = classification_report(y_test, y_pred)
print(report)