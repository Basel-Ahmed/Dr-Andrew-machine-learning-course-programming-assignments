function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%



cvalues=[0.01,0.1, 0.03, 0.1, 0.3, 1, 3, 10, 30 ];
sigmavalues=[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30 ];
minerror=2;
 %x11=Xval(:,1);
 %x22=Xval(:,2);
   errors=zeros(8,8);
for cvalue =1:length(cvalues)
    for sigmavalue=1:length(sigmavalues)
        
   
        %kernelFunction=gaussianKernel( Xval(:,1), Xval(:,2),sigma );
       
        %model=svmTrain(X   ,y  ,cvalues(cvalue)  ,  @(x11,x22)gaussianKernel( X(:,1)   , X(:,2)  ,sigmavalues(sigmavalue))) ;
        model=svmTrain(X   ,y  ,cvalues(cvalue)  ,  @(x11,x22)gaussianKernel( x11   , x22  ,sigmavalues(sigmavalue))) ;
        predictions=svmPredict(model,Xval);
        %errors(sigma,ic)=mean(double(predictions ~=yval));
        errors(sigmavalue,cvalue)=mean(double(predictions ~=yval));
        %if error<minerror
           % minerror=error;
            %C=cvalues(ic);
            %sigma=sigmavalues(sigmaa);
        %end
        
    end
    


  
end

%errors;

%[M,I] = min(errors(:));
%[row  ,  column]=ind2sub(size(errors),I);
%C=cvalues(column);
%C=C*10;
%sigma=sigmavalues(row);
%C
%sigma
[m,i]=min(min(errors));
cvalueindex=i;
C=cvalues(cvalueindex);
[M,I]=min(errors);
sigmaindex=I(cvalueindex);
sigma=sigmavalues(sigmaindex);




    



% =========================================================================

end
