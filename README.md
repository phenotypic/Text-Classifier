# Text-Classifier

This script trains a machine-learning model to classify text into categories. 

Once trained, the model can be used to predict categories for unseen text.

### Example use cases:

- Sentiment Analysis: classify texts into positive, negative, or neutral categories
- Spam Detection: categorize emails or messages as "spam" or "not spam"
- Topic Classification: categorise text documents into predefined topics or categories
- Language Detection: detect the language of a given text
- Censorship: detect and filter out inappropriate messages
- Authorship Attribution: determine the author of a text from a set of possible authors
- Intent Classification: determine what a user wants to achieve based on their input

## Usage

Clone the repository:
```
git clone https://github.com/phenotypic/Text-Classifier.git
```

Change to the project directory:
```
cd Text-Classifier
```

Install dependencies:
```
pip3 install -r requirements.txt
```

Run the script:
```
python3 classify.py
```

Here are some flags you can add:

| Flag | Description |
| --- | --- |
| `-m <model>` | Model: choose a model (defaults to `LogisticRegression`): `LogisticRegression`, `MultinomialNB`, `DecisionTreeClassifier`, `LinearSVC`, `RandomForestClassifier`, `GradientBoostingClassifier`, `KNeighborsClassifier`, `AdaBoostClassifier`, `MLPClassifier` |
| `-s <split>` | Split: define custom test set size (default: `0.2`) |

When training a new model, the script expects a CSV file in the same directory named `train.csv`. The file should contain two columns: `label` and `text`. The `label` for each `text` object can be binary or multiclass.

For example, if you were training a binary model for sentiment analysis, your CSV would look something like this (assuming `0` represents negative sentiment, and `1` represents positive sentiment):

```
label,text
1,"Really enjoyed this book, would definitely recommend"
0,"Such a boring read, avoid at all costs"
0,"Waste of money"
1,"10/10 excellent storyline"
1,"A friend recommended this to me - gripping book"
```

Alternatively, if you were training a multiclass model for authorship attribution with three different authors (`Alice`, `Bob`, `Charlie`), your CSV would look something like this:

```
label,text
Alice,"It was a blustery day, the leaves forming tempestuous maelstroms on the sidewalks, whirling in the cruel autumnal gusts"
Bob,"Kinda chilly today. Leaves everywhere, dancing around in the wind like nobody's business"
Charlie,"Autumn's harsh breath orchestrates a whirl of fallen foliage, conducting a symphony of seasonal decay on the city's walkways"
Bob,"Doesn't feel too warm out. The wind's got the leaves up in a fuss, skittering all over the pavement"
Alice,"The ferocity of the season declared itself in eddies of russet and amber, swirling relentlessly under the despotic rule of the wind"
```

Once the training data is loaded, the script will automatically split the file into training and testing sets.

After the classifier has been trained, it will be evaluated against the testing data and you will receive a final accuracy score:

```
Accuracy: 0.9291151659063925
```

Now that a model has been trained, it can be used to predict the categories for unseen text. For this, the script expects a CSV file in the same directory named `input.csv`. Using the authorship attribution example, the file might look something like this:

```
text
"Nighttime. Dark everywhere, and the moon and stars lighting up the sky"
"The moon painted a melancholy picture against the blackened canvas of the night"
"Under the shroud of night, the lunar orb weeps in celestial solitude, mourned by a retinue of distant suns"
"Not a lot of light, but the moon and stars are doing their best. They sure make the night kind of beautiful"
```

The script will then output the text alongside the model's predictions to a file named `predictions.csv`. For example:

```
text,prediction
"Nighttime. Dark everywhere, and the moon and stars lighting up the sky",Bob
"The moon painted a melancholy picture against the blackened canvas of the night",Alice
"Under the shroud of night, the lunar orb weeps in celestial solitude, mourned by a retinue of distant suns",Charlie
"Not a lot of light, but the moon and stars are doing their best. They sure make the night kind of beautiful",Bob
```
