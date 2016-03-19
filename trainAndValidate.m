function accuracy = trainAndValidate(hiddenUnits, lambda)
    %Read our x, y from the files
    [xtrain ytrain xval yval numberOfOutputLabels] = prepareDigitRecognizerDataForValidation();
    
    %Train the network based on training data
    fprintf("\nTraining the data on %d elements\n",size(xtrain,1));
    [Theta1, Theta2] = trainNetwork(xtrain, ytrain, hiddenUnits, numberOfOutputLabels, lambda);
    
    %Find predictions for validation data
    fprintf("\nFinding predictions on validation set\n");
    predictions = predict(Theta1, Theta2, xval);
    
    %Calculate the accuracy of predictions
    correctPredictions = sum(predictions == yval);
    accuracy = (correctPredictions/size(xval,1))*100;
    fprintf("I am able to get a accuracy of %f percent on the validation set\n", accuracy);
end