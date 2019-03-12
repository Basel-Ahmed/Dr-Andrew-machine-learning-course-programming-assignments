function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
n=size(theta);
t=2:1:n;
m = length(y); % number of training examples
j=zeros(m,1);
multiplicantfordrev=zeros(n);
regularizedparameters=theta(t,:);
%multiplicantfordrev=zeros(m,1);
multiplicantfordrev(t)=lambda/m;


% You need to return the following variables correctly 
J = 0;
grad = zeros(n);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

hypothesis=X*theta;
functionofhypothesis=sigmoid(hypothesis);
pos=find(y==1);
neg=find(y==0);
j(pos)=-log(functionofhypothesis(pos));
j(neg)=-log(1-functionofhypothesis(neg));
j=sum(j)/m;

squaredregularizedsum=regularizedparameters'*regularizedparameters;
regularizationterm=(squaredregularizedsum*lambda)/(2*m);

J=j+regularizationterm;
regularizationtermdrevative=multiplicantfordrev.*theta;
normalterm=((X')*(functionofhypothesis-y))/m;
grad=normalterm+regularizationtermdrevative;



% =============================================================

end
