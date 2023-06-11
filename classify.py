import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the data
df = pd.read_csv('input.csv')

# Vectorise the data
vector = TfidfVectorizer().fit_transform(df['text'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(vector, df['label'], test_size=0.2)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Test the model
predictions = model.predict(X_test)

# Print the accuracy
print('Accuracy:', accuracy_score(y_test, predictions))
