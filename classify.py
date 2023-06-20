import pandas, os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from joblib import dump, load


def train_model():
    # Load the data
    print('\nLoading data...')
    df = pandas.read_csv('train.csv')

    # Ensure columns are present and not empty
    assert 'text' in df.columns, 'Input file must contain a \'text\' column'
    assert 'label' in df.columns, 'Input file must contain a \'label\' column'
    assert df['text'].isna().sum() == 0, '\'text\' column should not have missing values'
    assert df['label'].isna().sum() == 0, '\'label\' column should not have missing values'

    print('\nTraining model...')
    # Vectorize the data
    vectorizer = TfidfVectorizer()
    vector = vectorizer.fit_transform(df['text'])

    # Split input data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(vector, df['label'], test_size=0.2)

    # Train the model
    model = LogisticRegression()
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
