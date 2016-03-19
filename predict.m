function p = predict(Theta1, Theta2, X)
% This function returns all predictions for X given Theta1 and Theta2
% Theta1 should be 30x785
m = size(X, 1);
num_labels = size(Theta2, 1);

% prediction function credit goes to the course assignment in Andrew Ng's Coursera course "Machine Learning"
p = zeros(size(X, 1), 1);

h1 = sigmoid([ones(m, 1) X] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');
[dummy, p] = max(h2, [], 2);
p(p == 10) = 0;
csvwrite('output_layer.txt', h2);

end