clc ;clear all; close all

%% ==================== Part 1: Email Preprocessing ====================

fprintf('\nPreprocessing sample email (emailSample1.txt)\n');

% Extract Features
file_contents = readFile('emailSample1.txt');
word_indices  = processEmail(file_contents);

% Print Stats
fprintf('Word Indices: \n');
fprintf(' %d', word_indices);
fprintf('\n\n');

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ==================== Part 2: Feature Extraction ====================

fprintf('\nExtracting features from sample email (emailSample1.txt)\n');

% Extract Features
file_contents = readFile('emailSample1.txt');
word_indices  = processEmail(file_contents);
features      = emailFeatures(word_indices);

% Print Stats
fprintf('Length of feature vector: %d\n', length(features));
fprintf('Number of non-zero entries: %d\n', sum(features > 0));

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =========== Part 3: Train Logistic Regression for Spam Classification ========

load('spamTrain.mat'); % Loading preprocessed data

m = size(X,1);
initial_theta = zeros(size(X,2),1);
lambda = 1;

options = optimset('GradObj', 'on', 'MaxIter', 100);

fprintf('Training data loaded \n');
fprintf('taking lambda = %f \n',lambda);
fprintf('Training the data...................... \n');

[theta, J, exit_flag] = ...
	fminunc(@(t)(CostFunctionReg(t, X, y, lambda)), initial_theta, options);

fprintf('training complete');

p = predict(theta, X);

fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
% Train accuracy = 99.925% 

%% =================== Part 4: Test Spam Classification ================
% After training the classifier, we can evaluate it on a test set.

load('spamTest.mat');

fprintf('\nEvaluating the trained logistic regression spam classifier on a test set ...\n')

p = predict(theta, Xtest);
fprintf('Test Accuracy: %f\n', mean(double(p == ytest)) * 100);
% Test accuracy is 99.1%




