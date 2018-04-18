wipe;

% create lscpd problem
size_tens = [1000 9 8 7]; R = 3;
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

% compute solution
[Uest,output] = lscpd_nls(A,b,U0,options);

% check error
frob(ful(U)-ful(Uest))/frob(ful(U))

% check some output stuff
output.cgiterations
output.iterations


