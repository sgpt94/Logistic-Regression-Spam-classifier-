function [J, grad] = costFunctionReg(theta, X, y, lambda)

m = length(y); % number of training examples

 
J = 0;
grad = zeros(size(theta));


z = X*theta;
h_x = sigmoid(z);
J = (-1/m)*sum(y.*log(h_x) + (1-y).*log(1-h_x)) + (lambda/(2*m))*sum(theta(2:end).^2);

grad(1) = (1/m)*X(:,1)'*(h_x - y);
grad(2:end) = (1/m)*(X(:,2:end)'*(h_x - y)) + (lambda/m)*theta(2:end);


end
