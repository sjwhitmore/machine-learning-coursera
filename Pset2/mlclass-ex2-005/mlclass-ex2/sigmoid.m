function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));
onesvector = ones(size(z));

g = onesvector./ (onesvector + e.^(-z));




% =============================================================

end
