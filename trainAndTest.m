function predictions = trainAndTest(hiddenUnits, lambda)
    
    %Read our x, y from the files
    [x y numberOfOutputLabels] = prepareDigitRecognizerDataForTesting();
    
    %Train the network based on training data
    [Theta1, Theta2] = trainNetwork(x, y, hiddenUnits, numberOfOutputLabels, lambda);
    
    % Read the test data
    xtest = csvread('test.csv');
    
    %Find predictions for test data
    predictions = predict(Theta1, Theta2, xtest);
    
    %Write the output to file
    csvwrite('predictions.txt', predictions);
end