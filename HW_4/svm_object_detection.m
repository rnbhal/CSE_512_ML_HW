function [arr_ap, arr_objval] = svm_object_detection(C)
%To start vifeat library
%startup();
load('trainAnno.mat');
[trD, trLb, valD, valLb, trRegs, valRegs] = HW4_Utils.getPosAndRandomNeg();
[w, bias, alpha, objective_function] = cquadprog(trD, trLb, C);


[f, n] = size(trD);
positive = [];
negative = [];
arr_objval = [];
arr_ap = [];
epsilon = 0.1;

for iter = 1:10
positive = [];
negative = [];
    for i = 1:size(trLb, 1)
       if trLb(i) == 1
           positive = [positive, trD(:, i)];
       else
           if alpha(i) < epsilon
               negative = [negative, trD(:, i)];
           end
       end
    end
HW4_Utils.genRsltFile(w, bias, "train", "train_result_file");
load("train_result_file.mat");
svm_violate = [];
for i = 1:length(rects)
    image = imread(sprintf('%s/trainIms/%04d.jpg', HW4_Utils.dataDir, i));
    [Height, Width,~] = size(image);
    current_rect = rects{i};
    current_rect = current_rect(:,and(current_rect(3,:) <= Width, current_rect(4,:) <= Height));
    ubs = ubAnno{i};
    size_ubs = size(ubs, 2);
    overlaps = [];
    limit_over = false;
    for j = 1:size_ubs
        ov_rect = HW4_Utils.rectOverlap(current_rect, ubs(:, j));
        overlaps = [overlaps, ov_rect];
    end        
    
    for j = 1:length(current_rect)
        if current_rect(5, j) > 0
           continue 
        end
        negative_example = 0;
        for k = 1:size_ubs
            if overlaps(j, k) > 0.35
                negative_example = 1;
                break;
            end
        end
        if negative_example == 0

            simage = image(int16(current_rect(2, j)):int16(current_rect(4, j)), int16(current_rect(1, j)):int16(current_rect(3, j)), :);
            simage = imresize(simage, HW4_Utils.normImSz);

            features = HW4_Utils.cmpFeat(rgb2gray(simage));
            features = features / norm(features);
            svm_violate = [svm_violate, features];

            if size(svm_violate, 2) > 1000
                limit_over = true;
                break;
            end
        end
        if limit_over == true
            break;
        end
    end
    if limit_over == true
        break;
    end
end
% size(negative)
% size(svm_violate)
negative = [negative, svm_violate];
trD = [];
trD = [trD, positive];
size_positive_labels = size(trD, 2);
trD = [trD, negative];

trLb = ones(size_positive_labels, 1);
negative_labels = -ones(size(negative, 2), 1);
trLb = [trLb; negative_labels];

% disp(size(trD));
% disp(size(trLb));

[w, bias, alpha, objval] = cquadprog(trD, trLb, C);
objval
arr_objval = [arr_objval, objval];

HW4_Utils.genRsltFile(w, bias, "val", "validation_result_file");
[ap, prec, rec] = HW4_Utils.cmpAP("validation_result_file", "val");
arr_ap = [arr_ap, ap];

end
numbers = linspace(1, 10, 10);
subplot(2,1,1);
plot(numbers, arr_objval);
subplot(2,1, 2);
plot(numbers, arr_ap);

HW4_Utils.genRsltFile(w, bias, "test", "112073893");
end