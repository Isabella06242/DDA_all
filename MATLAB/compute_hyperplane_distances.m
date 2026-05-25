function [distances,w,b] = compute_hyperplane_distances(X, y)
    % Compute distances from optimal separating hyperplane for binary classification
    %
    % Inputs:
    %   X - n x p matrix of features (n observations, p features)
    %   y - n x 1 vector of class labels (should be -1 and 1, or will be converted)
    %
    % Output:
    %   distances - n x 1 vector of signed distances from the hyperplane
    %               Positive values indicate correct side of hyperplane
    %               Negative values indicate wrong side of hyperplane
    
    % Ensure y contains only -1 and 1
    unique_labels = unique(y);
    if length(unique_labels) ~= 2
        error('Only binary classification is supported');
    end
    
    % Convert labels to -1 and 1 if necessary
    if ~all(ismember(y, [-1, 1]))
        y_converted = zeros(size(y));
        y_converted(y == unique_labels(1)) = -1;
        y_converted(y == unique_labels(2)) = 1;
        y = y_converted;
    end
    
    % Train linear SVM
    svm_model = fitcsvm(X, y, 'KernelFunction', 'linear', 'Standardize', true);
    
    % Extract hyperplane parameters
    % For linear SVM: decision function is f(x) = w'*x + b
    w = svm_model.Beta;  % Weight vector
    b = svm_model.Bias;  % Bias term
    
    % Compute signed distances
    % Distance = (w'*x + b) / ||w||
    % The sign indicates which side of the hyperplane
    decision_values = X * w + b;
    distances = decision_values / norm(w);
    
    % Make distances positive for correct classifications
    distances = distances .* y;
end