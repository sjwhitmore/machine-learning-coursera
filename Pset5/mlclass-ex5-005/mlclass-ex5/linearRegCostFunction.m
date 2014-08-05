function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

J = (1/(2*m))*sumsq((X*theta) - y) + (lambda/(2*m))*sumsq(theta(2:end,:));
grad = (1/m)*((X*theta - y)'*X)';
grad = grad.+(lambda/m)*[0; theta(2:end)];












% =========================================================================

grad = grad(:);

end
