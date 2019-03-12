function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
pos=[];
neg=[];
m = length(y); % number of training examples
j=zeros(m,1);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
%inea=zeros(m,1);
hypothesis=X*theta;
functionofhypothesis=sigmoid(hypothesis);
%partpos=y.*log10(h);
%partneg=(1-y).*(log10(1-h));
%partsum=-1*partpos-1*partneg;
%j=sum(partsum)/m;
pos=find(y==1);
neg=find(y==0);
j(pos)=-log(functionofhypothesis(pos));
j(neg)=-log(1-functionofhypothesis(neg));
J=sum(j);
J=J/m;
%J=-J;

%fprintf(J)
%fprintf('Cost at initial theta (zeros): %f\n',J);
grad=((X')*(functionofhypothesis-y))/m;
% =============================================================

end
