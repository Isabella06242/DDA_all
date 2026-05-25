function dydx = central_deriv(y, x, n)
    % CENTRALDERIVATIVE Computes central derivative approximation
    % 
    % Inputs:
    %   y - vector of function values
    %   x - vector of x values (same size as y)
    %   n - number of points to use (must be odd: 3, 5, 7, etc.)
    %
    % Output:
    %   dydx - vector of derivative approximations
    
    % Check inputs
    if nargin < 3
        n = 3; % Default to 3-point formula
    end
    
    if mod(n, 2) == 0
        error('Number of points must be odd');
    end
    
    if length(y) ~= length(x)
        error('x and y must have the same length');
    end
    
    if length(y) < n
        error('Input vector must have at least %d points', n);
    end
    
    % Ensure column vectors
    y = y(:);
    x = x(:);
    
    % Initialize output
    dydx = zeros(size(y));
    
    % Define coefficients for different orders
    switch n
        case 3
            % 3-point formula: f'(x) = (f(x+h) - f(x-h))/(2h)
            coeffs = [-1, 0, 1] / 2;
        case 5
            % 5-point formula: f'(x) = (f(x-2h) - 8f(x-h) + 8f(x+h) - f(x+2h))/(12h)
            coeffs = [1, -8, 0, 8, -1] / 12;
        case 7
            % 7-point formula
            coeffs = [-1, 9, -45, 0, 45, -9, 1] / 60;
        case 9
            % 9-point formula
            coeffs = [3, -32, 168, -672, 0, 672, -168, 32, -3] / 840;
        otherwise
            error('Formulas for n=%d points not implemented. Use 3, 5, 7, or 9.', n);
    end
    
    % Number of points on each side of center
    k = (n - 1) / 2;
    
    % Compute derivatives for interior points
    for i = (k+1):(length(y)-k)
        % Get the spacing (assuming uniform spacing locally)
        h = x(i+1) - x(i);
        
        % Apply the finite difference formula
        sum_val = 0;
        for j = -k:k
            sum_val = sum_val + coeffs(j+k+1) * y(i+j);
        end
        dydx(i) = sum_val / h;
    end
    
    % Handle boundary points using lower-order formulas
    % Left boundary
    for i = 1:k
        if i == 1
            % Forward difference for first point
            dydx(i) = (y(i+1) - y(i)) / (x(i+1) - x(i));
        else
            % Use 3-point centered difference where possible
            h = x(i+1) - x(i);
            dydx(i) = (y(i+1) - y(i-1)) / (2*h);
        end
    end
    
    % Right boundary
    for i = (length(y)-k+1):length(y)
        if i == length(y)
            % Backward difference for last point
            dydx(i) = (y(i) - y(i-1)) / (x(i) - x(i-1));
        else
            % Use 3-point centered difference where possible
            h = x(i) - x(i-1);
            dydx(i) = (y(i+1) - y(i-1)) / (2*h);
        end
    end
end