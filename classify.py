import pandas, os, argparse
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

classifiers = {
        'LogisticRegression': LogisticRegression(),
        'MultinomialNB': MultinomialNB(),
        'DecisionTreeClassifier': DecisionTreeClassifier(),
        'LinearSVC': LinearSVC(),
        'RandomForestClassifier': RandomForestClassifier(),
        'GradientBoostingClassifier': GradientBoostingClassifier(),
        'KNeighborsClassifier': KNeighborsClassifier(),
        'AdaBoostClassifier': AdaBoostClassifier(),
        'MLPClassifier': MLPClassifier()
    }

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, choices=list(classifiers.keys()), help='choose a model (default: LogisticRegression)', default='LogisticRegression')
parser.add_argument('-s', '--split', type=float, help='define custom test set size (default: 0.2)', default=0.2)
args = parser.parse_args()

def train_model():
    # Load the data
    print('\nLoading data...')
    df = pandas.read_csv('train.csv')

    # Ensure columns are present and not empty
    assert 'text' in df.columns, 'Input file must contain a \'text\' column'
    assert 'label' in df.columns, 'Input file must contain a \'label\' column'
    assert df['text'].isna().sum() == 0, '\'text\' column should not have missing values'
    assert df['label'].isna().sum() == 0, '\'label\' column should not have missing values'

    # Vectorize the data
    print('\nVectorizing data...')
    vectorizer = TfidfVectorizer()
    vector = vectorizer.fit_transform(df['text'])

    # Split data into training and testing sets
    print('\nSplitting data for training and testing...')
    x_train, x_test, y_train, y_test = train_test_split(vector, df['label'], test_size=args.split)

    # Train the model
    print(f'\nTraining model ({args.model})...')
    model = classifiers[args.model]
    model.fit(x_train, y_train)

    # Test the model
    predictions = model.predict(x_test)

    # Print evaluation metrics
    print('\nAccuracy:', accuracy_score(y_test, predictions))

    # Save the trained model and vectorizer
    dump(model, 'trained_model.joblib')
    dump(vectorizer, 'vectorizer.joblib')
    print('\nModel saved to trained_model.joblib')
    print('Vectorizer saved to vectorizer.joblib\n')


def predict():
    # Ensure model and vectorizer are trained and saved
    assert os.path.exists('trained_model.joblib'), 'Model is not trained yet'
    assert os.path.exists('vectorizer.joblib'), 'Vectorizer is not trained yet'

    # Load model and vectorizer
    print('\nLoading model...')
    model = load('trained_model.joblib')
    vectorizer = load('vectorizer.joblib')

    # Load prediction data
    print('\nLoading data...')
    df = pandas.read_csv('input.csv')

    # Ensure column is present and not empty
    assert 'text' in df.columns, 'Input file must contain a \'text\' column'
    assert df['text'].isna().sum() == 0, '\'text\' column should not have missing values'

    # Vectorize prediction data
    print('\nMaking predictions...')
    vector = vectorizer.transform(df['text'])

    # Make predictions
    predictions = model.predict(vector)

    # Save predictions
    df['prediction'] = predictions
    df.to_csv('predictions.csv', index=False)
    print('\nPredictions saved to predictions.csv\n')


def start_menu():
    print('\n1. Train model (train.csv required)')
    print('2. Make predictions (input.csv required)')
    option = input('\nChoose option: ')

    if option == '1':
        train_model()
    elif option == '2':
        predict()
    else:
        print('Invalid option')
        start_menu()

start_menu()
