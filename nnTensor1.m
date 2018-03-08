clearvars; close all;
rng(1);
% https://github.com/BorjaGIH/DeepTensor

%% Create feature dataset and multivariate output
numfeat = 4; % Number of features. numfeat+1 is the dimension(s) of the tensor
numpoints = 200; % Number of datapoints (each datapoint has numfeat values)
order = 4; % Order of the tensor. "order" is degree of the polynomial that tensor product achieves
lambda = 1e8; % Regularization parameter
X = randi(10,numpoints,numfeat); % Input data, numpoints x numfeat matrix

% Multiple Input Single Output (MISO)
for ii = 1:numpoints
    Ys(ii) = f1(X(ii,:)); % Output: univariate nonlinear combination of the inputs
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
regfactor = 1e-15;
W0 = regfactor*rand(initvec); % Dense random tensor

%% Optimization using kernel
kernel = Kernel1(Xtr,Ystr,lambda,numfeat,order); % create kernel
kernel.initialize(W0); % z0 is the initial guess for the variables, e.g., z0 = W0. lambda is the reg. parameter

options.TolFun = 1e-20; % optimization process options
options.MaxIter = 30;
% options.Display = 10;
% options.TolX = 1e-20;

% Minimize
[Wres,output] = minf_lbfgs(@kernel.objfun, @kernel.grad, W0, options); 
time = toc;

%% Tests
ErrTr = Ftest(Wres,Xtr,Ystr,order,numfeat); % in "train" data
figure
plot(ErrTr);
title('Error in train set (optimization of nonlinear function)')
disp(['Error norm divided by length in train data: ',num2str(norm(ErrTr/length(Xtr)))])

ErrTest = Ftest(Wres,Xte,Yste,order,numfeat); % in new data
figure
plot(ErrTest);
title('Error in test set (optimization of nonlinear function)')
disp(['Error norm divided by length in test data: ',num2str(norm(ErrTest(length(Xte))))])

%% Log file
if exist('log.txt', 'file') ~= 2 % when file does not exits
    fileID = fopen('log.txt','w');
    formatSpec = ' Rel. train err. || Rel. test err. || Time (s) || Iterations || Tensor order || Dimensions';
    fprintf(fileID,formatSpec);
    fclose(fileID);
    
    dimformat = string('%dx');
    for ii=1:length(size(Wres))-1
        dimformat = strcat(dimformat,'%dx');
    end
    fileID = fopen('log.txt','a+');
    formatSpec = strcat('\n %4.2f || %4.2f || %4.2f || %d || %4.2f || ',dimformat);
    fprintf(fileID,formatSpec,(norm(ErrTr)/length(Xtr)),(norm(ErrTest)/length(Xte)),time,output.iterations,order,size(Wres));
    fclose(fileID);
    
elseif exist('log.txt', 'file') == 2 % when file exists
    fileID = fopen('log.txt','a+');
    dimformat = string('%dx');
    for ii=1:length(size(Wres))-1
        dimformat = strcat(dimformat,'%dx');
    end
    formatSpec = strcat('\n %4.2f || %4.2f || %4.2f || %d || %4.2f || ',dimformat);
    fprintf(fileID,formatSpec,(norm(ErrTr)/length(Xtr)),(norm(ErrTest)/length(Xte)),time,output.iterations,order,size(Wres));
    fclose(fileID);
    
else % any other case
    disp('*** Error in writing file ***')
end
