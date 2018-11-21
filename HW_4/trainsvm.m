function [Accuracy, Obj, Conf, SupportVectors] = trainsvm(X, Y, W, B, C)
X = X';    
[n, t] = size(X);
output = sign((X*W) + B);
% size(output)
% YPredict = YPredict';
% 1. Accuracy 
Accuracy = sum(output == Y) / n;
WNorm = norm(W)^2;

% Objective Value of SVM
Obj = Y'*((X*W) + B);
Obj = 1 - Obj;
Obj = C * sum(Obj) + (WNorm/2);

% Number of Support Vector
YTemp = (X*W) + B;
[SupportVectors, ~] = size(YTemp(YTemp >= -1 & YTemp <= 1));

% Confusion Matrix
Conf = confusionmat(Y, output);
end
