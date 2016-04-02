function [Theta1, Theta2] = trainNetwork(x,y, hiddenUnits, numberOfOutputLabels, lambda)

% m is the total number of training data, and n is the total number of features/pixels
m = size(x,1);
n = size(x,2);

% Input layer size
inputLayerSize = n;

% Randomly initialize our Theta values. Add a bias to input layer and hidden layer
Theta1 = randInitializeWeights(inputLayerSize, hiddenUnits);
Theta2 = randInitializeWeights(hiddenUnits, numberOfOutputLabels);

% initialTheta is the combination of Theta1 and Theta2
initialTheta = [Theta1(:) ; Theta2(:)];

% Initialize lambda for regularization if it is not passed in the params
if( ! exist("lambda", "lambda"))
  lambda = 1;

% Getting ready for the training

% A reasonable value for max_iter
options = optimset('MaxIter', 100);

% Create "short hand" for the cost function to be minimized
costFunction = @(p) costAndGradient(p, ...
                                   x, y, lambda, ...
                                   hiddenUnits, ...
                                   numberOfOutputLabels);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[Theta, cost] = fmincg(costFunction, initialTheta, options);


% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(Theta(1:hiddenUnits * (inputLayerSize + 1)), ...
                 hiddenUnits, (inputLayerSize + 1));

Theta2 = reshape(Theta((1 + (hiddenUnits * (inputLayerSize + 1))):end), ...
                 numberOfOutputLabels, (hiddenUnits + 1));

end
