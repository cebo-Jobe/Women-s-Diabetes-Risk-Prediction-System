from DecisionTree import DecisionTree
import numpy as np
from collections import Counter
import joblib
import os


class RandomForest:
    def __init__(self, n_trees=300, max_depth=15, min_samples_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []
        self.feature_importances_ = None

    def fit(self, X, y):
        self.trees = []
        n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.feature_importances_ = np.zeros(X.shape[1])

        for _ in range(self.n_trees):
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_sample_split=self.min_samples_split,
                n_features=self.n_features
            )
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
            # Accumulate feature importances from all trees
            """Each tree's feature importance is typically based on:
               Features that separate patients well (high information gain) get more importance
               Features used on larger patient groups get more importance"""
            #Ensure the tree has computed feature importances before trying to access them
            if hasattr(tree, 'feature_importances_') and tree.feature_importances_ is not None:
                self.feature_importances_ += tree.feature_importances_

        # Normalize feature importances
        if np.sum(self.feature_importances_) > 0:
            self.feature_importances_ /= np.sum(self.feature_importances_)

    def save_model(self, filepath='models/random_forest_model.joblib'):
        """Save the trained model to disk using joblib

        Args:
            filepath (str): Path where to save the model
        """
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save the entire model object
        joblib.dump(self, filepath) #Serializes the Python object to disk
        print(f"Model saved to {filepath}")

    @staticmethod
    def load_model(filepath='models/random_forest_model.joblib'):
        """Load a trained model from disk using joblib

        Args:
            filepath (str): Path to the saved model

        Returns:
            RandomForest: Loaded model object
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found at {filepath}")

        model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return model

    #This introduces variance between trees, reducing overfitting when predictions are averaged.
    #Each tree trains on a bootstrap sample of data.
    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True) #Indices and size
        return X[idxs], y[idxs]

    def predict(self, X):
        if X.ndim == 1:  # Check if the input array 'X' has only one dimension.
            X = X.reshape(1, -1)

        predictions = np.array([tree.predict(X) for tree in self.trees])
        #prediction across multiple trees
        tree_preds = np.swapaxes(predictions, 0, 1)
        #Majority Voting for Each Patient
        predictions = np.array([self._most_common_label(pred) for pred in tree_preds])
        return predictions

    def predict_proba(self, X):
        """Return prediction probabilities for each class"""
        if X.ndim == 1:
            X = X.reshape(1, -1)

        predictions = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(predictions, 0, 1)

        probabilities = []
        for pred in tree_preds:
            counter = Counter(pred) #labe: frequency
            total = len(pred) #(number of trees)
            prob_0 = counter.get(0, 0) / total
            prob_1 = counter.get(1, 0) / total
            probabilities.append([prob_0, prob_1])

        return np.array(probabilities)

    def predict_risk_percentage(self, X):
        """Return diabetes risk as percentage and risk category

        Args:
            X: Feature matrix for prediction

        Returns:
            tuple: (risk_percentages, risk_categories, binary_predictions)
                - risk_percentages: Array of risk percentages (0-100)
                - risk_categories: Array of risk category strings ('LOW RISK' or 'HIGH RISK')
                - binary_predictions: Array of binary predictions (0 or 1)
        """
        probabilities = self.predict_proba(X) #2D array, sample by prob_percent [0.6,0.3]

        # Get probability of diabetes (class 1)
        diabetes_probabilities = probabilities[:, 1]

        # Convert to percentages
        risk_percentages = diabetes_probabilities * 100

        # Categorize risk levels
        risk_categories = []
        binary_predictions = []

        for risk_pct in risk_percentages:
            if risk_pct < 40:
                risk_categories.append('LOW RISK')
                binary_predictions.append(0)
            else:
                risk_categories.append('HIGH RISK')
                binary_predictions.append(1)

        return risk_percentages, np.array(risk_categories), np.array(binary_predictions)

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def print_random_forest(self, max_trees=None, feature_names=None):
        """Print all trees in the random forest (or a subset).

        Args:
            max_trees: Maximum number of trees to print (None for all)
            feature_names: Optional list of feature names for better readability
        """
        if max_trees is None:
            max_trees = len(self.trees)

        print(f"\nRandom Forest with {len(self.trees)} trees (showing {max_trees}):")
        print(f"Max depth: {self.max_depth}")
        print(f"Min samples split: {self.min_samples_split}")
        print(f"Number of features: {self.n_features}\n")

        for i, tree in enumerate(self.trees[:max_trees]):
            print(f"\n=== Tree {i + 1} ===")
            tree.print_tree(feature_names=feature_names)
            print("=" * 20)

    def confusion_matrix(self, y_true, y_pred):
        """Compute confusion matrix for binary classification.

        Args:
            y_true: Array of true labels (0 or 1)
            y_pred: Array of predicted labels (0 or 1)

        Returns:
            A 2x2 numpy array: [[TN, FP], [FN, TP]]
        """
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return np.array([[tn, fp], [fn, tp]])

    def accuracy_score(self, y_true, y_pred):
        """Compute accuracy score.

        Args:
            y_true: Array of true labels
            y_pred: Array of predicted labels

        Returns:
            Accuracy score (float between 0 and 1)
        """
        return np.mean(y_true == y_pred)

    def precision_score(self, y_true, y_pred):
        """Compute precision score.

        Args:
            y_true: Array of true labels (0 or 1)
            y_pred: Array of predicted labels (0 or 1)

        Returns:
            Precision score (float between 0 and 1)
        """
        cm = self.confusion_matrix(y_true, y_pred)
        tp = cm[1, 1]
        fp = cm[0, 1]
        return tp / (tp + fp) if (tp + fp) > 0 else 0

    def recall_score(self, y_true, y_pred):
        """Compute recall score (sensitivity).

        Args:
            y_true: Array of true labels (0 or 1)
            y_pred: Array of predicted labels (0 or 1)

        Returns:
            Recall score (float between 0 and 1)
        """
        cm = self.confusion_matrix(y_true, y_pred)
        tp = cm[1, 1]
        fn = cm[1, 0]
        return tp / (tp + fn) if (tp + fn) > 0 else 0

    def f1_score(self, y_true, y_pred):
        """Compute F1 score (harmonic mean of precision and recall).

        Args:
            y_true: Array of true labels (0 or 1)
            y_pred: Array of predicted labels (0 or 1)

        Returns:
            F1 score (float between 0 and 1)
        """
        precision = self.precision_score(y_true, y_pred)
        recall = self.recall_score(y_true, y_pred)
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    def evaluate(self, X, y):
        """Compute and print all evaluation metrics.

        Args:
            X: Feature matrix
            y: True labels

        Returns:
            Dictionary containing all metrics
        """
        y_pred = self.predict(X)

        metrics = {
            'accuracy': self.accuracy_score(y, y_pred),
            'precision': self.precision_score(y, y_pred),
            'recall': self.recall_score(y, y_pred),
            'f1': self.f1_score(y, y_pred),
            'confusion_matrix': self.confusion_matrix(y, y_pred)
        }

        print("Evaluation Metrics:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print("Confusion Matrix:")
        print(metrics['confusion_matrix'])

        return metrics