# Text-Classifier

This script trains a machine learning model to classify text into categories.

Example use cases:

- Sentiment Analysis: classify texts into positive, negative, or neutral categories
- Spam Detection: categorize emails or messages as "spam" or "not spam"
- Topic Classification: categorise text documents into predefined topics or categories
- Language Detection: detect the language of a given text
- Censorship: detect and filter out inappropriate messages
- Authorship Attribution: determine the author of a text from a set of possible authors
- Intent Classification: determine what a user wants to achieve based on their input

## Usage

Download with:
```
git clone https://github.com/phenotypic/Text-Classifier.git
pip3 install -r requirements.txt
```

Run from the same directory with:
```
python3 classify.py
```

The script expects a CSV file in the same directory named `input.csv`.

The CSV file should contain the columns `label` and `text`. For example, if you were training a classifier for sentiment analysis, your CSV would look something like this (assuming `0` represents negative sentiment, and `1` represents positive sentiment):

```
label,text
1,"Really enjoyed this book, would definitely recommend
0,"Such a boring read, avoid at all costs"
0,"Waste of money"
1,"10/10 excellent storyline"
1,"A friend recommended this to me - gripping book"
```

The script will automatically split the CSV file into training (80%) and testing (20%) data sets.

After running the script, the classifier will be trained on the training data using the Logistic Regression model. It will then be evaluated against the testing data, and you will receive an accuracy score:

```
Accuracy: 0.9191151659063925
```
