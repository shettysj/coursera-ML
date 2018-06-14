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

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.





h_theta = X * theta;

tot_cost= sum((h_theta - y).^2);
tot_theta= sum(theta(2:length(theta)).^2);

J= ((1/(2*m)) * tot_cost) + ((lambda/(2*m)) * tot_theta);

%gradient%
cost_grad= (1/m) * (X' * (h_theta - y));
theta_grad= theta(2:length(theta));


grad(1)= cost_grad(1);
grad(2:length(theta))= cost_grad(2:length(theta)) + ((lambda/m) * theta_grad);





% =========================================================================

grad = grad(:);

end
