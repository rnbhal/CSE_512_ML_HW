function [trainK, testK] = cmpExpX2Kernel(trainD, testD, gamma, TestRequired)
            [n, d] = size(trainD);
            [t, f] = size(testD);
            p = 0;
            trainK = zeros(n,n);
            for i=1:n
                for j=1:n
                num = ((trainD(i,:)-trainD(j,:)).^2);
                den = (trainD(i,:)+trainD(j,:)) + eps('single');
                trainK(i,j) = sum(num./den);
                den = -1/gamma;
                trainK(i,j) = exp(trainK(i,j)*den);
                end
            end
            trainK = [(1:n).', trainK];
            testK = zeros(t,n);
            if TestRequired
                for i=1:t
                    for j=1:n
                    num = ((testD(i,:)-trainD(j,:)).^2);
                    deno = (testD(i,:)+trainD(j,:)) + eps('single');
                    testK(i,j) = sum(num./deno);
                    deno = -1/gamma;
                    testK(i,j) = (testK(i,j)*deno);
                    end
                end
            end
            testK = [(1:t).', testK];
           
        end