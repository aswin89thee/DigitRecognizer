# DigitRecognizer
Feed forward neural network solution to the "Digit Recognizer" [Kaggle competition](https://www.kaggle.com/c/digit-recognizer).

This is a 3-layer neural network. With 300 hidden elements, this script was able to achieve a test set accuracy of 94.02%. With 400 hidden elements, the test set accuracy is 94.27%.

Implemented in Octave (Open source Matlab alternative)

Huge thanks to Andrew Ng's [Machine Learning course](www.ml-class.org). A lot of code here have been taken from the course's assignments and modifying them slightly for this purpose.

To train your network and predict the output for the test set, supply the number of hidden layer neurons and the regularization parameter as follows:
                            predictions = trainAndTest(hiddenUnits, lambda);
This outputs the test predictions in "predictions.txt".

To divide your training data into training and validation set, and test the validation accuracy, run it like this:
                            accuracy = trainAndValidate(hiddenUnits, lambda);
