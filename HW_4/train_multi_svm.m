function [Accuracy] = multisvm(X, Y, W, B, C)
X = X';    
[n, t] = size(X);
output = [];

for i=1:n
    b_temp = 0;
    k = 1;
    for j=1:10
        temp = X(i,:)*W(:,j) + B(:,j);
        if temp > b_temp
            b_temp = temp;
            k = j;
        end
    end
    output = [output, k];
end
output = output';

% 1. Accuracy 
Accuracy = sum(output == Y) / n;
end