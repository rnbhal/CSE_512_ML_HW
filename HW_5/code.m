g = [0.1 0.3, 0.4780, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2 12 15];

% g = [0.1];
% g = [0.01 0.1 1, 10, 20, 40, 80, 100];
c = [0.1, 1, 10, 20, 40, 80,160, 200];
% g = [1 3 6 8 10 12 15 18 21 24 27];
% c = [0.01 0.05 0.1 0.3 0.6 1 10 100];
for i=1:length(g)
    [trDK, tstDK] = HW5_BoW.cmpExpX2Kernel(trD, tstD, g(i), false);
    for j=1:length(c)
        disp(sprintf('g %f   c %f',g(i),c(j)));
        model = svmtrain(trLbs, trDK, sprintf('-t 4 -c %f -g %f -v 5 -q',c(j),g(i)));
%         model = svmtrain(trainLabel, trDK, sprintf('-t 4 -c %f -g %f -q',c(j),g(i)));
%         [~,acc,~] = svmpredict(testLabel, tstDK, model);
    end
    disp(sprintf('\n'));
end

c=20
g=1.4

[trDK, tstDK] = HW5_BoW.cmpExpX2Kernel(trD, tstD, g, true);
model = svmtrain(trLbs, trDK, sprintf('-t 4 -c %f -v 5 -g %f -q',c,g));
model = svmtrain(trLbs, trDK, sprintf('-t 4 -c %f -g %f -q',c,g));
[output,acc,~] = svmpredict(zeros(1600,1), tstDK, model);
output = [tstIds , output];
csvwrite('predTestLabels.csv', output);