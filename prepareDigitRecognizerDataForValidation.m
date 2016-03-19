function [xtrain ytrain xval yval numberOfOutputLabels] = prepareDigitRecognizerDataForValidation()

 x = csvread('train.csv');
  
% Clear the labels row
x = x(2:size(x,1),:);

% Since the first row is the actual digit, this is our y vector
y = x(:,1);

% The rest of the columns are the pixel values. This becomes our x matrix
x = x(:,2:size(x,2));

% Replace all 0's as 10s. This makes it easier to work with matrix indexes
y(y==0) = 10;

% Output labels
numberOfOutputLabels = 10;

% Total number of elements
m = size(x,1);

% Lets use 60% of our test set for training, and 40% for validation
trainM = ceil(m*0.6);
xtrain = x(1:trainM, :);
ytrain = y(1:trainM, :);
xval = x(trainM+1:m, :);
yval = y(trainM+1:m, :);
%fprintf("Size of xtrain is %d x %d, Size of ytrain is %d x %d\n",size(xtrain),size(ytrain));
%fprintf("Size of xval is %d x %d, Size of yval is %d x %d\n",size(xval),size(yval));

end