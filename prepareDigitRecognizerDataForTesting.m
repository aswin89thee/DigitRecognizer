function [x y numberOfOutputLabels] = prepareDigitRecognizerDataForTesting()

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

end