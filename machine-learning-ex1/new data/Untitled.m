
data = csvread('housingdata.csv');
m = size(data,1);
traineddataul=ceil(m*0.6);
X = data(1:traineddataul, 1:13);
y = data(1:traineddataul, 14);
validdatall=traineddataul+1;
validdataul=traineddataul+0.2*m;
%fprintf('%f',m)
Xval=data(validdatall:validdataul,1:13);
yval=data(validdatall:validdataul,14);
testdatall=validdataul+1;
testdataul=m;
Xtest=data(testdatall:testdataul,1:13);
ytest=data(testdatall:testdataul,14);
save('mydata.mat','X','Xval','Xtest','y','yval','ytest')

fprintf(' x = [%.0f %.0f], y = %.0f \n', [Xval(1:10  ,  :) yval(1:10 )]');