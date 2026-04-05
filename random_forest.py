import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def evaluate_random_forest(data_path, n_trees=10, depth=10):
    print(f"Loading preprocessed data from {data_path}...")
    df = pd.read_csv(data_path)

    # Separate features (X) and target (y)
    X = df.drop(columns=['Painting_Target'])
    y = df['Painting_Target']

    # Split into 80% training and 20% validation
    # 'stratify=y' ensures all three paintings are equally represented in both sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"\nTraining Random Forest with {n_trees} trees and max depth of {depth}...")

    # Initialize the Random Forest model
    rf_model = RandomForestClassifier(
        n_estimators=n_trees,
        max_depth=depth,
        random_state=42,
        max_features=10



        ,
    )

    # Train the model
    rf_model.fit(X_train, y_train)

    # Generate predictions
    train_preds = rf_model.predict(X_train)
    val_preds = rf_model.predict(X_val)

    # Calculate accuracy
    train_acc = accuracy_score(y_train, train_preds)
    val_acc = accuracy_score(y_val, val_preds)

    print("\n--- Model Performance ---")
    print(f"Training Accuracy:   {train_acc * 100:.2f}%")
    print(f"Validation Accuracy: {val_acc * 100:.2f}%")

    # Check for overfitting
    if (train_acc - val_acc) > 0.15:
        print("\nWARNING: The model might be overfitting! The training accuracy is significantly higher than the validation accuracy.")
    else:
        print("\nThe model appears to be generalizing well to unseen data.")

    return rf_model

if __name__ == "__main__":
    # You can tweak your hyperparameters right here!
    model = evaluate_random_forest('processed_ml_dataset.csv', n_trees=50, depth=20)
