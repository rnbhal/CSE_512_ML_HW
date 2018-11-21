function [arr_ss, arr_p1, arr_p2, arr_p3] = kmeans(trX, trY, k, detailed)

[n, f] = size(trX);
k_center = randi(255,k,f);

iters = (1:10);
arr_ss = [];
arr_p1 = [];
arr_p2 = [];
arr_p3 = [];
iter = k;
% for iter=1:10

    for i=1:iter
        k_center(i,:) = trX(i,:);
    end
    xcenter = [];
    sso = 0;
    for i=1:20
        xcenter = findclosetcenter(trX, k_center);
        [k_center, ssn] = adjustcenter(k_center, xcenter, trX, iter, false);
        if sso == ssn
            disp(sprintf('   %d round',i));
            break;
        end
        sso = ssn;
    end
    sso
    arr_ss = [arr_ss sso];
    if detailed
        p1 = 0;
        p2 = 0;
        p1n = 0;
        p2n = 0;
        for i=1:n
            for j=i+1:n
                if trY(i,:) == trY(j,:)
                    p1n = p1n+1;
                    if xcenter(i,:) == xcenter(j,:)
                        p1 = p1+1;
                    end
                else
                    p2n = p2n+1;
                    if xcenter(i,:) ~= xcenter(j,:)
                        p2 = p2+1;
                    end
                end
            end
        end

        % p1
        % p2
        % N
        p1 = p1/p1n
        p2 = p2/p2n
        p3 = (p1+p2)/2
        arr_p1 = [arr_p1 p1];
        arr_p2 = [arr_p2 p2];
        arr_p3 = [arr_p3 p3];
    end
% end

% figure(1)
% plot(iters,arr_ss, '-o');
% title('Plot Sum of Squares vs k')
% xlabel('K')
% ylabel('Sum of Squares')
% legend('Sum of Squares')
% 
% figure(2)
% plot(iters,arr_p1,'-o', 'Color', [1, 0, 0])
% hold on
% plot(iters, arr_p2,'-o', 'Color', [0, 1, 0])
% hold on
% plot(iters, arr_p3,'-o', 'Color', [0, 0, 1])
% title('Plot P1 P2 P3 vs k')
% xlabel('K')
% ylabel('P1 P2 P3')
% legend('P1','P2', 'P3')

end