clearvars; close all;
rng(1);
% https://github.com/BorjaGIH/DeepTensor
% LS_CPD branch

%% Create feature dataset and multivariate output
numfeat = 4;             % Number of features. numfeat+1 is the dimension(s) of the tensor
numpoints = 100;         % Number of datapoints (each datapoint has numfeat values)
order = 3;               % Order of the tensor. "order" is degree of the polynomial that tensor product achieves
lambda = 1e8;            % Regularization parameter
X = randi(10,numpoints,numfeat); % Input data, numpoints x numfeat matrix
rank = 3;                % rank of the tensor, for constraint/efficient representation
options.MaxIter = 200;   % optimization iterations
nonlin = true;           % learned function is nonlinear. 1 or 0

% Multiple Input Single Output (MISO)
for ii = 1:numpoints
    Ys(ii) = f1(X(ii,:),nonlin); % Output: univariate nonlinear combination of the inputs
end
Ys = Ys';

% Divide dataset for train and test
sizetr = 0.7;
sizete = 0.3;

Xtr = X(1:sizetr*numpoints,:);
Ystr = Ys(1:sizetr*numpoints); % MISO

Xte = X(1:sizete*numpoints,:);
Yste = Ys(1:sizete*numpoints);

%% Create tensor and find "good" initial point
% Order of the tensor: numfeat. Dimensions: numfeat 
tic % time
initvec = repmat(numfeat+1,1,order);
regfactor = 1e-6;
W0 = regfactor*rand(initvec); % Dense random tensor
% To do: find good initial point

%% Create initial U0
tic % time
size_tens = [numpoints numfeat+1 numfeat+1 numfeat+1];
U0 = cpd_rnd(size_tens(2:end),rank);

%% Optimization using kernel
% [U0,~] = cpd(W0,rank); % factorize and ensure we take a rank smaller than that estimated by rankest
% % R = rankest(W0);
% % if rank>R
% %     rank = R;
% %     disp(['Inputed rank is bigger than estimated by rankest, so it has been changed '])
% % end
% 
% kernel = Kernel1(Xtr,Ystr,lambda,numfeat,order,rank); % create kernel
% kernel.initialize(U0); % z0 is the initial guess for the variables, e.g., z0 = W0. lambda is the reg. parameter
% 
% %%%%%%%% Notes %%%%%%%%
% % Rank of the reconstructed solution tensor is 1, instead of "rank"
% % Check if this is normal, seeing that the optimization variables are the
% % factor matrices.
% 
% % some optimization process options
% options.TolFun = 1e-20; 
% % options.Display = 10;
% % options.TolX = 1e-20;
% 
% optimizer = 'minf_lbfgsdl'; % this must coincide with the used function
% [Ures,output] = minf_lbfgsdl(@kernel.objfun, @kernel.grad, U0, options); % Minimize
% Wres = cpdgen(Ures); % reconstruct tensor from factors
% time = toc;

%% Optimization LS-CPD
optimizer = 'ls-cpd nls';
A = coeffMatrix(numfeat, order, Xtr); % build A coefficient matrix
b = Ystr;

% Solver options
options.Display = true;
options.TolFun = eps^2;
options.TolX = eps;
% options.CGMaxIter = prod(size_tens(2:end));

% compute solution
[Ures,output] = lscpd_nls(A,b,U0,options);

% check error
% frob(ful(U)-ful(Uest))/frob(ful(U)

Wres = cpdgen(Ures);
time = toc;

%% Tests
ErrTr = Ftest(Wres,Xtr,Ystr,order,numfeat); % in "train" data
figure
plot(ErrTr);
title('Error in train set (optimization of nonlinear function)')
disp(['Error norm divided by length in train data: ',num2str(norm(ErrTr/length(Xtr)))])

ErrTest = Ftest(Wres,Xte,Yste,order,numfeat); % in "new" data
figure
plot(ErrTest);
title('Error in test set (optimization of nonlinear function)')
disp(['Error norm divided by length in test data: ',num2str(norm(ErrTest/length(Xte)))])

%% Log file
if exist('log.txt', 'file') ~= 2 % when file does not exist
    fileID = fopen('log.txt','w');
    formatSpec = ' Rel. train err. || Rel. test err. || Time (s) || Iterations || Stop info || Tensor order || Dimensions || Rank || Nonlin. f || Optimizer ||';
    fprintf(fileID,formatSpec);
    fclose(fileID);
    
    dimformat = string('%dx');
    for ii=1:length(size(Wres))-1
        dimformat = strcat(dimformat,'%dx');
    end
    
    fileID = fopen('log.txt','a+');
    formatSpec = strcat('\n %4.2f || %4.2f || %4.2f || %d || %d || %d || ',dimformat, ' || %d || %d || %10s');
    fprintf(fileID,formatSpec,(norm(ErrTr)/length(Xtr)),(norm(ErrTest)/length(Xte)),time,output.iterations,output.info,order,size(Wres),rank,nonlin,optimizer);
    fclose(fileID);
    
elseif exist('log.txt', 'file') == 2 % when file exists
    fileID = fopen('log.txt','a+');
    
    % if I still want to write header when file exists...
%     formatSpec = '\n Rel. train err. || Rel. test err. || Time (s) || Iterations|| Stop info || Tensor order || Dimensions || Rank || Nonlin. f || Optimizer';
%     fprintf(fileID,formatSpec);
%     fclose(fileID);
    
    dimformat = string('%dx');
    for ii=1:length(size(Wres))-1
        dimformat = strcat(dimformat,'%dx');
    end
    
    fileID = fopen('log.txt','a+');
    formatSpec = strcat('\n %4.2f || %4.2f || %4.2f || %d || %d || %d || ',dimformat, ' || %d || %d || %10s');
    fprintf(fileID,formatSpec,(norm(ErrTr)/length(Xtr)),(norm(ErrTest)/length(Xte)),time,output.iterations,output.info,order,size(Wres),rank,nonlin,optimizer);
    fclose(fileID);
    
else % any other case
    disp('*** Error in writing file ***')
end
