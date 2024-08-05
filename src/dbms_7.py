import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sqlalchemy import create_engine
import warnings
import logging as log

# Ignore all warnings
warnings.filterwarnings('ignore')

try:
    train_data = pd.read_csv('./data/train.csv')
    test_data = pd.read_csv('./data/test.csv')

    def handle_missing_values(data):
        data['Age'].fillna(data['Age'].median(), inplace=True)
        data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
        if 'Fare' in data.columns:
            data['Fare'].fillna(data['Fare'].median(), inplace=True)
        data.drop(columns=['Cabin'], inplace=True)
        return data

    def feature_engineering(data):
        data['Title'] = data['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
        data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
        return data

    def load_data_from_db(engine, table_name):
        query = f"SELECT * FROM {table_name}"
        data_from_db = pd.read_sql(query, engine)
        return data_from_db

    train_data = handle_missing_values(train_data)
    test_data = handle_missing_values(test_data)

    train_data = feature_engineering(train_data)
    test_data = feature_engineering(test_data)

    numerical_cols = ['Age', 'Fare', 'FamilySize']
    categorical_cols = ['Sex', 'Embarked', 'Title']

    # Define the transformers
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # Create a column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    # Fit and transform the data
    train_data_transformed=preprocessor.fit_transform(train_data)
    test_data_transformed=preprocessor.transform(test_data)

    # Get the transformed column names
    transformed_columns = numerical_cols + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols))

    # Convert the transformed data back to DataFrames for convenience
    train_transformed_df = pd.DataFrame(train_data_transformed, columns=transformed_columns)
    test_transformed_df = pd.DataFrame(test_data_transformed, columns=transformed_columns)

    # Create an SQLite database engine
    engine = create_engine('sqlite:///titanic.db')

    # Save the transformed DataFrame to tables
    train_transformed_df.to_sql('titanic_train_transformed', engine, index=False, if_exists='replace')
    test_transformed_df.to_sql('titanic_test_transformed', engine, index=False, if_exists='replace')

    train_loaded = load_data_from_db(engine, 'titanic_train_transformed')
    test_loaded = load_data_from_db(engine, 'titanic_test_transformed')

    # Separate the target from features
    X_train = train_loaded
    y_train = train_data['Survived']

    # Split the training data into training and validation sets
    X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    pipeline = Pipeline(steps=[('model', LogisticRegression(max_iter=1000))])

    # Train the model
    pipeline.fit(X_train_split, y_train_split)

    # Make predictions on the validation set
    y_val_pred = pipeline.predict(X_val)

    # Evaluate the model
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Validation Accuracy: {val_accuracy:.2f}")

    # Make predictions on the test set
    test_pred = pipeline.predict(test_loaded)

    # If the test data does not have 'Survived', we can't calculate test accuracy
    if 'Survived' in test_data.columns:
        y_test = test_data['Survived']
        test_accuracy = accuracy_score(y_test, test_pred)
        print(f"Test Accuracy: {test_accuracy:.2f}")
    else:
        print("Test predictions completed. 'Survived' column not found in test data to calculate accuracy.")
except Exception as e:
    log.error(f"Error Message:{e}")
