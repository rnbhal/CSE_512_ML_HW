function [weights, bias] = multiclass(X, Y, C)
[t,n] = size(X);
weights = [];
bias = [];
for k=1:10
    fprintf('\nRound %d, training classifier\n', k);
    yk = Y;
    for i = 1:n
        if yk(i) == k
            yk(i) = 1.0;
        else
            yk(i) = -1.0;
        end
    end
    [w, b, Obj] = cquadprog(X, yk, C);
    [Accuracy, Obj, Conf, SupportVectors] = trainsvm(X, yk, w, b, C);
    Accuracy
    weights = [weights, w];
    bias = [bias, b];
end

end