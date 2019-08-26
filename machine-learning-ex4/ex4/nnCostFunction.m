function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% % ********* PART 1:  Forward propagation **********

% % disp("size of y"), size(y)
% % disp("size of X"), size(X)
% % disp("size of Theta1"), size(Theta1)
% % disp("size of Theta2"), size(Theta2)
% % disp("size of hidden layer"), hidden_layer_size
% % disp("num_labels"), num_labels


% % Create vector for the output labels
ylabels = 1:num_labels;

% % Create matrix Y with m rows and num_labels column
Y = zeros(m, num_labels);
for i = 1:size(Y, 1)
    Y(i,:) = (ylabels == y(i,:)); 
endfor

% % **** Layer 2 ****
% % Add bias units to input data
A1 = [ones(size(X,1),1) X];
 
% % Compute activation units for 2nd (hidden) layer
Z2 = A1 * Theta1';
H2 = sigmoid(Z2);

% % **** (Output) Layer 3 ****
A2 = [ones(size(Z2,1), 1) H2]; 
H3 = A3 =  sigmoid(A2 * Theta2');

J = (1/m)*sum(sum((-Y).*log(H3) - (1-Y).*log(1 - H3), 2));

% % ********* Cost computation for regularization terms **********
 
% Remove the bias columns
theta1 = Theta1(:,2:end);
theta2 = Theta2(:,2:end);

% Total regularization term cost
total_reg = (lambda /(2 * m)) * (sum(sum(theta1 .^ 2, 2)) + sum(sum(theta2 .^ 2, 2)));

% % ********* Total cost function **********
J = J + total_reg;
% J;


% % ********* PART 2:  Backpropagation **********

% disp("size of y"), size(y)
% disp("size of H3"), size(H3)
% disp("size of H2"), size(H2)
% disp("size of Theta2"), size(Theta2)
% disp("size of Theta1"), size(Theta1)

% % Step 1
% A1 = X(:, 2:end);
% disp("size of A1"), size(A1)

% % Step2:  Layer 3 (output layer)
% Sigma3 = zeros(size(H3));
% for i = 1 : size(H3, 1)
%     y_i = (ylabels == y(i,:));
%     Sigma3(i,:) = H3(i,:) - y_i;
% endfor

% disp("size of Sigma3"), size(Sigma3)
% size(sigmoidGradient(A1 * theta1'))

% % Step 3:  Layer 2 (hidden layer)
% Sigma2 = (Sigma3 * theta2) .* sigmoidGradient(A1 * theta1');
% disp("size of Sigma2"), size(Sigma2)

% Delta1 = Sigma2' * X(:, 2:end);
% Delta2 = Sigma3' * H2(:, 2:end);

% Theta1_grad = Delta1 ./ m;
% Theta2_grad = Delta2 ./ m;

% disp("Theta1_grad"), size(Theta1_grad)
% disp("Theta2_grad"), size(Theta2_grad)





% *********************************
% *********************************
% I = eye(num_labels);
% Y = zeros(m, num_labels);
% for i=1:m
%   Y(i, :)= I(y(i), :);
% end



% A1 = [ones(m, 1) X];
% Z2 = A1 * Theta1';
% A2 = [ones(size(Z2, 1), 1) sigmoid(Z2)];
% Z3 = A2*Theta2';
% H = A3 = sigmoid(Z3);


% penalty = (lambda/(2*m))*(sum(sum(Theta1(:, 2:end).^2, 2)) + sum(sum(Theta2(:,2:end).^2, 2)));

% J = (1/m)*sum(sum((-Y).*log(H) - (1-Y).*log(1-H), 2));
% J = J + penalty;

Sigma3 = A3 - Y;
Sigma2 = (Sigma3*Theta2 .* sigmoidGradient([ones(size(Z2, 1), 1) Z2]))(:, 2:end);


Delta_1 = Sigma2'*A1;
Delta_2 = Sigma3'*A2;


Theta1_grad = Delta_1./m + (lambda/m)*[zeros(size(Theta1,1), 1) Theta1(:, 2:end)];
Theta2_grad = Delta_2./m + (lambda/m)*[zeros(size(Theta2,1), 1) Theta2(:, 2:end)];



% % =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];



end
