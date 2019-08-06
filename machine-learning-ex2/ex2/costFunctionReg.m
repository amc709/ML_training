function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


% This is theta without theta[0]
theta2 = theta(2:size(theta, 1), :);

sig = sigmoid(X * theta);

% Cost
J = (1/m) * (-y' * log(sig) - (1-y)'*log(1-sig)) + (lambda/(2*m)) * (theta2' * theta);

% Gradient
grad0 = (1/m) * ((sig - y)' * X(:,1));
gradj = (1/m) * ((sig - y)' * X(:, 2:size(X,2)))' + (lambda/m) * theta(2:size(theta,1), :);
grad = [grad0; gradj];


% =============================================================

end