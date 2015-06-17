function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    params = length(theta);
    theta_temp = zeros(params,1);
    
    for i = 1:params
      sum = 0.0;
      for j = 1:m
        sum = sum + (X(j,:)*theta - y(j))*X(j,i);
      endfor
      theta_temp(i) = theta(i) - (alpha/m)*sum;
    endfor
      
    for i = 1:params
      theta(i) = theta_temp(i);
    endfor
    
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
