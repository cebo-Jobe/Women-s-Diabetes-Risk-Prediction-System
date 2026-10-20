
import numpy as np
from collections import Counter

from imblearn.over_sampling import SMOTE


class DataSplitter:
    def __init__(self, test_size=0.2, random_state=None, stratify=False):
        """
        Custom data splitting class to replace sklearn's train_test_split

        Parameters:
        - test_size: float (0.0 to 1.0), proportion of dataset to include in test split
        - random_state: int, random seed for reproducibility
        - stratify: bool, whether to preserve class distribution in splits
        """
        self.test_size = test_size
        self.random_state = random_state
        self.stratify = stratify

    def split(self, X, y):
        """
        Split arrays into random train and test subsets

        Returns:
        - X_train, X_test, y_train, y_test
        """
        if self.random_state is not None: #Ensures reproducible results
            np.random.seed(self.random_state)

        n_samples = len(X)
        test_samples = int(n_samples * self.test_size)

        if self.stratify:
            return self._stratified_split(X, y, test_samples)
        else:
            return self._random_split(X, y, test_samples)

    def _random_split(self, X, y, test_samples):
        """Random split without stratification
        Args:
            X: Feature matrix
            y: Target labels
            test_samples: Number of samples to allocate to test set
        Returns:
            X_train, X_test, y_train, y_test: Split datasets
        """
        # Create array of indices [0, 1, 2, ..., n_samples-1]
        indices = np.arange(len(X))

        # Shuffle indices randomly to ensure random split
        np.random.shuffle(indices)

        # Split indices into test and train portions
        test_idx = indices[:test_samples]  # First test_samples go to test set
        train_idx = indices[test_samples:]  # Remaining go to train set

        # Return split data using the indices
        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

    #This function splits data into training and test sets while preserving the original class proportions
    def _stratified_split(self, X, y, test_samples):
        """Stratified split preserving class distribution
        Args:
            X: Feature matrix
            y: Target labels
            test_samples: Number of samples to allocate to test set
        Returns:
            X_train, X_test, y_train, y_test: Split datasets with preserved class ratios
        """
        # Get unique classes and their counts in the dataset
        classes = np.unique(y)
        class_counts = Counter(y)

        # Initialize empty lists to store indices
        train_indices = []
        test_indices = []

        # Process each class separately
        for cls in classes:
            # Get indices of all samples belonging to current class
            cls_indices = np.where(y == cls)[0]

            # Shuffle class indices to ensure randomness
            np.random.shuffle(cls_indices)

            # Calculate how many test samples this class should contribute
            # Proportional to class's representation in full dataset
            n_test = int(test_samples * (class_counts[cls] / len(y)))

            # Add first n_test samples to test set
            test_indices.extend(cls_indices[:n_test])
            # Add remaining samples to train set
            train_indices.extend(cls_indices[n_test:])

        # Shuffle final indices to avoid ordered patterns
        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)

        # Return split data using the stratified indices
        return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

class ImbalanceHandler:
    def __init__(self, strategy='smote', random_state=None):
        """
        Handles class imbalance in the training dataset.

        Parameters:
        - strategy: str, 'smote' or 'ros' (RandomOverSampler)
        - random_state: int, for reproducibility
        """
        self.strategy = strategy
        self.random_state = random_state
        self.sampler = None


    def fit_resample(self, X_train, y_train):
        """
        for imbalance data model might achieve high accuracy but poor recall
        Balances the training data so machine learning model doesn't get biased toward the majority class.
        Resamples the training data to balance the classes.

        Returns:
        - X_train_resampled, y_train_resampled
        """
        # Check if we even need to resample
        class_counts = Counter(y_train)
        if len(class_counts) < 2 or min(class_counts.values()) == max(class_counts.values()):
            print("Classes are balanced. No resampling applied.")
            return X_train, y_train

        print(f"Class distribution before resampling: {class_counts}")

        # Choose the resampling strategy
        if self.strategy == 'smote':
            # Creates diverse samples, reduces overfitting
            #mechanisim: new_point = sample + random(0,1) * (neighbor - sample)
            self.sampler = SMOTE(random_state=self.random_state)
        elif self.strategy == 'ros': # Can cause overfitting (exact duplicates), but faster
            from imblearn.over_sampling import RandomOverSampler
            self.sampler = RandomOverSampler(random_state=self.random_state)
        else:
            raise ValueError("Strategy must be 'smote' or 'ros'")

        X_res, y_res = self.sampler.fit_resample(X_train, y_train)
        print(f"Class distribution after resampling: {Counter(y_res)}")
        return X_res, y_res