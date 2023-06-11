# Text-Classifier

This script is a simple implementation of a text classifier which leverages machine learning.

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

The CSV file should contain the columns `label` and `text`. For example, if you were training a model for sentiment analysis, your CSV would look something like this (assuming `0` represents negative sentiment, and `1` represents positive sentiment):

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
