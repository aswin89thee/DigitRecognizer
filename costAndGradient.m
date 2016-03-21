function [J, grad] = costAndGradient(Theta, X, y, lambda, hiddenLayerSize, outputLabels)

% fprintf('\n Inside costAndGradient, size of Theta is (%d,%d)\n', size(Theta));
input_layer_size = size(X,2);
m = size(X, 1);
grad = zeros(size(Theta));
%Reshaping the Theta parameters
Theta1 = reshape(Theta(1:hiddenLayerSize * (input_layer_size + 1)), ...
                 hiddenLayerSize, (input_layer_size + 1));

Theta2 = reshape(Theta((1 + (hiddenLayerSize * (input_layer_size + 1))):end), ...
                 outputLabels, (hiddenLayerSize + 1));

% We want y to have outputLabels number of columns, with the index of y value as 1 and everything else 0
expandedY = zeros(m, outputLabels);
for i=1:m
  expandedY(i,y(i)) = 1;
end

h1 = sigmoid([ones(m, 1) X] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');
hx = h2;
oneMinusHx = 1 .- hx;

J = expandedY .* log(hx) + (1 - expandedY) .* log(oneMinusHx);
J = (-1/m) * sum(sum(J));
% fprintf('\nCost for this Theta is %f\n', J);

% Regularized cost. Note that you shouldn't square the first column of theta parameters
regularizedTheta1 = Theta1(:,2:size(Theta1,2));
regularizedTheta2 = Theta2(:,2:size(Theta2,2));
regularizedThetaSum = (lambda/(2*m)) * ( sum(sum((regularizedTheta1 .^ 2))) + sum(sum((regularizedTheta2 .^ 2))));
J = J + regularizedThetaSum;

% Compute gradients using backpropagation
DELTA1 = zeros(size(Theta1));
DELTA2 = zeros(size(Theta2));
A2 = h1;
A3 = h2;

% Vectorized backprop begin
thetaonex = Theta1 * [ones(m,1) X]';
z2 = [ones(m,1) thetaonex'];
siggradz2 = sigmoid(z2) .* (1 - sigmoid(z2));
delta3 = A3 - expandedY;
delta2 = (Theta2' * delta3') .* siggradz2';
DELTA2 = delta3' * [ones(m,1) A2];
DELTA1 = delta2(2:end,:) * [ones(m,1) X];
% Vectorized backprop end

% Non-vectorized backprop begin
% for i = 1:m
    % Compute/get the layers
%     a1 = X(i,:);
%     a2 = A2(i,:);
%     a3 = A3(i,:);
%     curExpandedY = expandedY(i,:);
%     
    % Computing z value
%     thetaOneX = Theta1 * [1 a1]';
%     z2 = [1 ; thetaOneX];
%     sigGradz2 = sigmoid(z2) .* (1 - sigmoid(z2));
%     
%     %Compute delta and DELTA
%     delta3 = a3 - curExpandedY;
%     delta2 = (Theta2' * delta3') .* sigGradz2;
%     DELTA2 = DELTA2 + (delta3' * [1 a2]);
%     DELTA1 = DELTA1 + (delta2(2:end) * [1 a1]);
% end
% Non-vectorized backprop end

% Divide the overall gradient by m to get the gradient for one example
Theta1_grad = DELTA1/m;
Theta2_grad = DELTA2/m;

% Regularize the gradients
regularizationParam1 = (lambda/m) * Theta1;
regularizationParam1(:,1) = 0;
Theta1_grad = Theta1_grad + regularizationParam1;

regularizationParam2 = (lambda/m) * Theta2;
regularizationParam2(:,1) = 0;
Theta2_grad = Theta2_grad + regularizationParam2;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end