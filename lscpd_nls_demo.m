% wipe;
clearvars; close all; clc

% create lscpd problem
N = 3;
dim = 10;
R = 2;
M = (dim*N-N+1)*R+1;
size_tens = [1000 repmat(dim,1,N)];
U = cpd_rnd(size_tens(2:end),R);
def = 98;
A = [rand(2000,dim^N-def),repmat(rand(2000,1),1,def)];
size(A)
rank(A)
b = A*tens2vec(ful(U));


% Borja's tests. Perturb A or b (so that b-bCheck is of order as in
% nnTensor1.m (1e-13)
factor = 1e-10;
b2 = b + factor*rand(size(b));
frob(b-b2);
A2 = A + factor*rand(size(A));
frob(b-A2*tens2vec(cpdgen(U)));

% set options for nls solver
% options.Display = true;
options.TolFun = eps^2;
options.TolX = eps;
options.CGMaxIter = prod(size_tens(2:end));

% set initial solution
U0 = cpd_rnd(size_tens(2:end),R);

% compute solution
[Uest,output] = lscpd_nls(A2,b2,U0,options);

% check error
frob(ful(U)-ful(Uest))/frob(ful(U))

% check some output stuff
% output.cgiterations
% output.iterations


