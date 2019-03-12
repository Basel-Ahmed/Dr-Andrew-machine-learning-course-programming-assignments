feature1=1:1:22;
feature2=1:1:22;
y=[1,2,3,3,3,2,2,2,1,1,1,1,2,2,3,3,2,1,2,1,3,3 ];

indiciesofyequalto1=find(y==1);
indiciesofyequalto2=find(y==2);
indiciesofyequalto3=find(y==3);
plot(feature1(indiciesofyequalto1),feature2(indiciesofyequalto1),  'g*')%  , 'LineWidth', 2   , 'MarkerSize'  , 7)
hold on
plot(feature1(indiciesofyequalto2),feature2(indiciesofyequalto2) , 'c*  ')%,'LineWidth', 2  ,   'MarkerSize'    ,  7)
hold on
plot(feature1(indiciesofyequalto3),feature2(indiciesofyequalto3) , 'r+') %, 'LineWidth',    2  ,    'MarkerSize'   ,   7)
x_point=[1 22]; %min point % max point
%hypothesis=theta0+x-point* theta1
theta1=2;
theta0=1;
plot(x_point  ,   theta1*[1 22]+theta0) % decision boundary