function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%


j=X*Theta'-Y;
j=j(R==1);
j=sum(power(j,2));
regularizedterm1=sum(sum(power(Theta,2)));
regularizedterm2=sum(sum(power(X,2)))
J=0.5*j+ 0.5*lambda*(regularizedterm1+regularizedterm2);



for i=1:size(Theta,1)
    idx = find(R(:, i)==1);
    temp= (Theta(i,:)*  (X( idx  , :  )  )'  )'-Y( idx , i  );
    Theta_grad(i,:)=temp(:,1)'  *X( idx , :) +lambda*Theta(i,:)   ;
    
end
for k=1:size(X,1)
    %idx1 = find(R(i, 1)==1);
    idx1 = find(R(k, :)==1);
    Thetatemp = Theta(idx1, :);
    Ytemp = Y(k, idx1);
    X_grad(k, :) = (X(k, :) *Thetatemp'  -Ytemp)*Thetatemp   +lambda*X(k,:) ;
   %  X_grad(k,:)= ( X(k,:)*(Theta(R(k,:)==1,: )')-Y(R(k,:)==1 ,  : )) *  (Theta( R(k,: )==1, :)  );
%end

%Theta_grad(:);
 %X_grad(:);








% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
