function [k_center,SS] = adjustcenter(oldk_center, xcenter, trX, k, display)

[k,f] = size(oldk_center);
k_center = randi(255, k, f);
SS = 0;
for i=1:k
    kth = trX(xcenter==i, :);
    [n, f] = size(kth);
    k_center(i,:) = sum(kth)/n;
    for j=1:n
        SS = SS + sum( (kth(j,:)-k_center(i,:)) .^ 2);
    end
end
if display
    disp(SS);
end;

end
