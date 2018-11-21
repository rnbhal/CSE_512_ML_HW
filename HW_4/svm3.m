function [ap,prec, rec] = svm3()
%Uncomment this for using Quadratic SVM
[trD, trLb, valD, valLb, trRegs, valRegs] = HW4_Utils.getPosAndRandomNeg();
[w, b, ~] = cquadprog(trD, trLb, 10);
HW4_Utils.genRsltFile(w, b, "val", "resultFile");
[ap, prec, rec] = HW4_Utils.cmpAP("resultFile", "val");

end