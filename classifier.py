import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional

def compute_hyperplane_distances(X: np.ndarray, y: np.ndarray, 
                                standardize: bool = True) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute distances from optimal separating hyperplane for binary classification
    
    Parameters:
    -----------
    X : np.ndarray
        n x p matrix of features (n observations, p features)
    y : np.ndarray
        n x 1 vector of class labels (will be converted to -1 and 1)
    standardize : bool, optional
        Whether to standardize features before training (default: True)
        
    Returns:
    --------
    distances : np.ndarray
        n x 1 vector of signed distances from the hyperplane
        Positive values indicate correct side of hyperplane
        Negative values indicate wrong side of hyperplane
    w : np.ndarray
        Weight vector of the hyperplane
    b : float
        Bias term of the hyperplane
        
    Raises:
    -------
    ValueError: If not binary classification
    """
    
    # Ensure inputs are numpy arrays
    X = np.asarray(X)
    y = np.asarray(y).flatten()  # Ensure y is 1D
    
    # Check for binary classification
    unique_labels = np.unique(y)
    if len(unique_labels) != 2:
        raise ValueError(f'Only binary classification is supported. Found {len(unique_labels)} classes.')
    
    # Convert labels to -1 and 1
    # Map the first unique label to -1, second to 1
    y_binary = np.where(y == unique_labels[0], -1, 1)
    
    # Standardize features if requested
    if standardize:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X
        scaler = None
    
    # Train linear SVM
    svm_model = SVC(kernel='linear', C=1.0)
    svm_model.fit(X_scaled, y_binary)
    
    # Extract hyperplane parameters
    # For linear SVM: decision function is f(x) = w'*x + b
    w = svm_model.coef_.flatten()  # Weight vector (p x 1)
    b = svm_model.intercept_[0]    # Bias term
    
    # Compute decision values (w'*x + b)
    decision_values = X_scaled @ w + b
    
    # Compute signed distances from hyperplane
    # Distance = (w'*x + b) / ||w||
    w_norm = np.linalg.norm(w)
    if w_norm > 0:
        distances = decision_values / w_norm
    else:
        distances = decision_values
    
    # Alternative: Get signed distances directly from decision function
    # distances = svm_model.decision_function(X_scaled) / w_norm
    
    # Make distances positive for correct classifications
    # This matches MATLAB's behavior: distance should be positive when classified correctly
    signed_distances = distances * y_binary
    
    return signed_distances, w, b, svm_model, scaler


def predict_with_hyperplane(X_test: np.ndarray, w: np.ndarray, b: float, 
                           scaler: Optional[StandardScaler] = None) -> np.ndarray:
    """
    Predict using pre-trained hyperplane parameters
    
    Parameters:
    -----------
    X_test : np.ndarray
        Test data (n x p)
    w : np.ndarray
        Weight vector from training
    b : float
        Bias term from training
    scaler : StandardScaler, optional
        Scaler used during training (if standardization was applied)
        
    Returns:
    --------
    predictions : np.ndarray
        Predicted labels (-1 or 1)
    distances : np.ndarray
        Signed distances from hyperplane
    """
    
    X_test = np.asarray(X_test)
    
    # Apply same standardization if scaler was used
    if scaler is not None:
        X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled = X_test
    
    # Compute decision values
    decision_values = X_test_scaled @ w + b
    
    # Predict labels (-1 or 1)
    predictions = np.sign(decision_values)
    
    # Compute distances
    w_norm = np.linalg.norm(w)
    if w_norm > 0:
        distances = decision_values / w_norm
    else:
        distances = decision_values
    
    return predictions, distances


def evaluate_classification(X_train: np.ndarray, y_train: np.ndarray, 
                           X_test: np.ndarray, y_test: np.ndarray,
                           standardize: bool = True) -> dict:
    """
    Complete classification pipeline with evaluation
    
    Parameters:
    -----------
    X_train, y_train : Training data and labels
    X_test, y_test : Test data and labels
    standardize : Whether to standardize features
    
    Returns:
    --------
    results : dict with all classification metrics and parameters
    """
    
    # Convert test labels to -1 and 1 based on training label mapping
    unique_labels = np.unique(y_train)
    y_train_binary = np.where(y_train == unique_labels[0], -1, 1)
    y_test_binary = np.where(y_test == unique_labels[0], -1, 1)
    
    # Train SVM and get distances
    train_distances, w, b, svm_model, scaler = compute_hyperplane_distances(
        X_train, y_train, standardize=standardize
    )
    
    # Predict on test set
    test_predictions, test_distances = predict_with_hyperplane(X_test, w, b, scaler)
    
    # Calculate accuracy
    train_accuracy = np.mean((train_distances > 0).astype(int))
    test_accuracy = np.mean(test_predictions == y_test_binary)
    
    # Calculate margin (minimum distance from hyperplane for correctly classified points)
    train_margin = np.min(train_distances[train_distances > 0]) if np.any(train_distances > 0) else 0
    test_margin = np.min(test_distances[test_predictions == y_test_binary]) if np.any(test_predictions == y_test_binary) else 0
    
    # Compile results
    results = {
        'weights': w,
        'bias': b,
        'svm_model': svm_model,
        'scaler': scaler,
        'train_distances': train_distances,
        'test_distances': test_distances,
        'test_predictions': test_predictions,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'train_margin': train_margin,
        'test_margin': test_margin,
        'label_mapping': {unique_labels[0]: -1, unique_labels[1]: 1},
        'support_vectors': svm_model.support_vectors_ if hasattr(svm_model, 'support_vectors_') else None
    }
    
    return results


# Example usage:
if __name__ == "__main__":
    # Generate example data
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    
    # Create two classes with different means
    X_class1 = np.random.randn(n_samples//2, n_features) - 1
    X_class2 = np.random.randn(n_samples//2, n_features) + 1
    X = np.vstack([X_class1, X_class2])
    
    # Create labels (can be any two distinct values)
    y = np.array([0] * (n_samples//2) + [1] * (n_samples//2))
    
    # Split into train/test
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Run classification
    results = evaluate_classification(X_train, y_train, X_test, y_test)
    
    print("Classification Results:")
    print(f"Training accuracy: {results['train_accuracy']:.3f}")
    print(f"Testing accuracy: {results['test_accuracy']:.3f}")
    print(f"Number of features: {len(results['weights'])}")
    print(f"Weight vector shape: {results['weights'].shape}")
    print(f"Bias term: {results['bias']:.4f}")
    
    # Show some predictions
    print("\nFirst 10 test predictions:")
    for i in range(min(10, len(y_test))):
        pred = "Class 1" if results['test_predictions'][i] == -1 else "Class 2"
        actual = "Class 1" if y_test[i] == np.unique(y_train)[0] else "Class 2"
        distance = results['test_distances'][i]
        print(f"Sample {i+1}: Predicted={pred}, Actual={actual}, Distance={distance:.4f}")