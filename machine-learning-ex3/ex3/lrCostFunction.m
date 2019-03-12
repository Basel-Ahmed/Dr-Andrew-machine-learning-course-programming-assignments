function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n=size(X,2);
%pos=find(y==1);
%neg=find(y==0);
%eachelementcost=zeros(m,1);
%allelementscost=0;

pos=[];
neg=[];
%m = length(y); % number of training examples
j=zeros(m,1);

n=size(theta);
t=2:1:n;
%m = length(y); % number of training examples
%j=zeros(m,1);
multiplicantfordrev=zeros(n);
regularizedparameters=theta(t,:);
%multiplicantfordrev=zeros(m,1);
multiplicantfordrev(t)=lambda/m;


% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));



% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

hypothesis=X*theta;
functionofhypothesis=sigmoid(hypothesis);

pos=find(y==1);
neg=find(y==0);
j(pos)=-log(functionofhypothesis(pos));
j(neg)=-log(1-functionofhypothesis(neg));
j=sum(j)/m;
%J=sum(j);
%J=J/m;
squaredregularizedsum=regularizedparameters'*regularizedparameters;
regularizationterm=(squaredregularizedsum*lambda)/(2*m);

J=j+regularizationterm;
regularizationtermdrevative=multiplicantfordrev.*theta;
normalterm=((X')*(functionofhypothesis-y))/m;
grad=normalterm+regularizationtermdrevative;


%eachelementcost(pos)=log(sigmoid((X(pos,:))*theta));
%eachelementcost(neg)=log(1-sigmoid((X(neg,:))*theta));
%allelementscost=sum(eachelementcost);
%J=-allelementscost/m;











% =============================================================

grad = grad(:);

end
