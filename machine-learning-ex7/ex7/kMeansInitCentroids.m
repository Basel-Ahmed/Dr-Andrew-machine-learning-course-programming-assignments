function centroids = kMeansInitCentroids(X, K)
%KMEANSINITCENTROIDS This function initializes K centroids that are to be 
%used in K-Means on the dataset X
%   centroids = KMEANSINITCENTROIDS(X, K) returns K initial centroids to be
%   used with the K-Means on the dataset X
%

% You should return this values correctly
centroids = zeros(K, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should set centroids to randomly chosen examples from
%               the dataset X
%

%p = randperm(n) returns a row vector containing a random permutation of the integers from 1 to n inclusive.

%p = randperm(n,k) returns a row vector containing k unique integers selected randomly from 1 to n inclusive.
randix=randperm(size(X,1),K);
centroids=X(randix,:);





% =============================================================

end

