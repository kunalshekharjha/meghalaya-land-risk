import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train():
    print("Loading data...")
    # 1. Load Data
    try:
        df = pd.read_csv('data/raw/synthetic_meghalaya_land_data.csv')
    except FileNotFoundError:
        print("ERROR: Data file not found! Did you run make_dataset.py?")
        return

    # 2. Preprocessing
    print("Preprocessing data...")
    df_clean = pd.get_dummies(df, columns=['district', 'land_type'], drop_first=True)
    
    # FIX: Correctly map 'No Dispute' to 0 and 'Disputed' to 1
    # The previous code looked for 'None', which doesn't exist anymore.
    df_clean['is_disputed'] = df_clean['dispute_status'].apply(lambda x: 1 if x == 'Disputed' else 0)
    
    # Define Features (X) and Target (y)
    X = df_clean.drop(['land_id', 'dispute_status', 'is_disputed'], axis=1)
    y = df_clean['is_disputed']
    
    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Train Model
    print("Training Random Forest Model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 5. Evaluate
    print("Evaluating...")
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"\nModel Training Complete!")
    print(f"Accuracy: {accuracy:.2%}")
    print("-" * 30)
    print("Detailed Report:")
    print(classification_report(y_test, predictions))

if __name__ == "__main__":
    train()