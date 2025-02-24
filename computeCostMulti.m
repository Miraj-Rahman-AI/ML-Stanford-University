function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% Compute the hypothesis
h = X * theta;

% Compute the errors
errors = h - y;

% Compute the cost
J = (1 / (2 * m)) * sum(errors .^ 2);

% =========================================================================

end