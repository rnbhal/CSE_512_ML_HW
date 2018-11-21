function [w, b, alpha, Obj] = cquadprog(X, Y, C)
X=X';
kernel = X*X';
kernel = double(kernel);
[n,m] = size(X);
dy = diag(Y);
H = dy*kernel*dy;
H = double(H);
f = -ones(n,1);
aeq = Y';
aeq = double(aeq);
beq = zeros(1,1);
lb = zeros(n,1);
ub = C*ones(n,1);
a = [];
b = [];
%quad programming using matlab
% size(Y)
% size(f)
% size(aeq)
% size(beq)
[alpha,fval] = quadprog(H,f,a,b,aeq,beq,lb,ub);
% max(alpha)
% min(alpha)
% size(alpha)
Obj = fval;
w = (diag(alpha)*Y)'*X;
w = w';
b = [];
b = Y - (X*w);
b = mean(b);
bestAcc = 0.0;
fb = b;
fa = 0;
end