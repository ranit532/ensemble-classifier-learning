import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, VotingClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import mlflow
import mlflow.sklearn
from src.data_loader import load_data
from src.utils import save_model, save_plot

def train_and_evaluate(model, X_train, y_train, X_test, y_test, model_name):
    """Trains a model, evaluates it, and saves the model and a confusion matrix plot."""
    with mlflow.start_run(run_name=model_name):
        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{model_name} Accuracy: {accuracy:.2f}")

        # Log metrics to MLflow
        mlflow.log_metric("accuracy", accuracy)

        # Create and save the confusion matrix plot
        cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(ax=ax)
        plt.title(f"{model_name} Confusion Matrix")
        save_plot(fig, f"{model_name}_confusion_matrix")
        plt.close(fig)

        # Save the model
        save_model(model, model_name)
        mlflow.sklearn.log_model(model, model_name)

def main():
    # Load and preprocess data
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # --- Model Definitions ---
    # Base estimator for Bagging and Extra Trees
    base_estimator = DecisionTreeClassifier(random_state=42)

    # 1. Bagging Classifier
    bagging_clf = BaggingClassifier(estimator=base_estimator, n_estimators=50, random_state=42)

    # 2. Extra Trees Classifier
    extra_trees_clf = ExtraTreesClassifier(n_estimators=50, random_state=42)

    # 3. Voting Classifier (Hard Voting)
    clf1 = LogisticRegression(random_state=42)
    clf2 = DecisionTreeClassifier(random_state=42)
    clf3 = SVC(probability=True, random_state=42)
    voting_clf = VotingClassifier(estimators=[('lr', clf1), ('dt', clf2), ('svc', clf3)], voting='hard')

    # 4. Stacking Classifier
    estimators = [
        ('lr', LogisticRegression(random_state=42)),
        ('svc', SVC(probability=True, random_state=42))
    ]
    stacking_clf = StackingClassifier(estimators=estimators, final_estimator=DecisionTreeClassifier())


    # --- Training and Evaluation ---
    mlflow.set_experiment("Ensemble Classifiers")

    train_and_evaluate(bagging_clf, X_train, y_train, X_test, y_test, "Bagging")
    train_and_evaluate(extra_trees_clf, X_train, y_train, X_test, y_test, "ExtraTrees")
    train_and_evaluate(voting_clf, X_train, y_train, X_test, y_test, "VotingClassifier")
    train_and_evaluate(stacking_clf, X_train, y_train, X_test, y_test, "Stacking")

if __name__ == "__main__":
    main()
