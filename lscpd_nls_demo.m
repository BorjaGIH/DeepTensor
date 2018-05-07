% wipe;
clearvars; close all; clc

% create lscpd problem
size_tens = [15 9 8 7]; R = 3;
U = cpd_rnd(size_tens(2:end),R);
A = randn(size_tens(1),prod(size_tens(2:end)));
b = A*tens2vec(ful(U));

% set options for nls solver
options.Display = true;
options.TolFun = eps^2;
options.TolX = eps;
options.CGMaxIter = prod(size_tens(2:end));

% set initial solution
U0 = cpd_rnd(size_tens(2:end),R);

% divide train and test
Atrain = A(1:10,:);
Atest = A(11:end,:);
bTrain = b(1:10);
bTest = b(11:end);

% compute solution
[Uest,output] = lscpd_nls(Atrain,bTrain,U0,options);

% check performance of the "classifier" in the train data
bEstim = Atrain*tens2vec(ful(Uest));
errorTrain = norm(bTrain-bEstim)/norm(bTrain)
 
% check performance of the classifier in new data
bTestEstim = Atest*tens2vec(ful(Uest));
errorTest = norm(bTest-bTestEstim)/norm(bTest)

% check error
frob(ful(U)-ful(Uest))/frob(ful(U))

% check some output stuff
% output.cgiterations
% output.iterations
