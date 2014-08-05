function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
for i = 1:length(y)
	J = J - (1/m)*(y(i).*log(sigmoid(X(i,:)*theta))+(1-y(i)).*(log(1-sigmoid(X(i,:)*theta))));
	%disp((1/m)*(y(i).*log(sigmoid(X(i,:)*theta))+(1-y(i)).*(log(1-sigmoid(X(i,:)*theta)))));
	grad = grad + ((1/m)*(sigmoid(X(i,:)*theta) - y(i))*(X(i,:)))';
end








% =============================================================

end
