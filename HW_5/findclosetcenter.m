function [x_center] = findclosetcenter(X, k_center)

[n, f] = size(X);
x_center = zeros(n,1);

for i=1:n
    best_SS = 100000000;
    for id = 1:size(k_center,1)
        SS = sum((k_center(id,:)- X(i,:)).^ 2);
        if best_SS > SS
            x_center(i,:) = id;
            best_SS = SS;
        end
    end
end
end