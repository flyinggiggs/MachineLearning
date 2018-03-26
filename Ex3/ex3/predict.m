function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% X has zie 5000,400
% theta1 has size 25,401
% theta2 has size 10,26

xOne = [ones(m, 1) X ];

% layer2 is 5000,25
layer2 = xOne * Theta1';

% add bias unit a0(5000,26)
layer2 = [ones(m, 1), sigmoid(layer2)];

% get layer3 values (5000,10) and find the maximum value's index
[max_result, max_index] = max(sigmoid(layer2 * Theta2'), [], 2);
  
p = max_index;



% =========================================================================


end
