function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
X=[ones(m,1)  X];
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

z1=Theta1*X';%calculate linear combination of hidden layer units for each training example % give matrix  of size (25,5000)
%each column represent  linear combination for one training example

a1=sigmoid(z1); %activation functionfor allunits for all training examples 


% add new new row to a1 to include bias at the top of the matrix
a1=[ones(1,m) ; a1];   % size become (26,5000)

z2=Theta2*a1;

a2=sigmoid(z2);


%probabilities of each classifier for each trainig example 
prob=a2';
[maxvaleo,maxindex]=max(prob,[],2);

p=maxindex;








% =========================================================================


end
