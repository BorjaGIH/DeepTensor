clearvars; close all;
rng(1);
% https://github.com/BorjaGIH/DeepTensor
% LS_CPD branch

%% Create feature dataset and output
numfeat = 9;             % Number of features. numfeat+1 is the dimension(s) of the tensor
numpoints = 150;         % Number of datapoints (each datapoint has numfeat values)
order = 4;               % Order of the tensor. "order" is degree of the polynomial that tensor product achieves
X = 4*rand(numpoints,numfeat); % Input data, numpoints x numfeat matrix
X = [ones(numpoints,1) X]; % Add bias term
rank = 3;                % Rank of the tensor, for constraint/efficient representation
factor = 1e-4;           % factor to multiply tensor, for "good" initial point (temporary). SNR = abs(exponent)

size_tens = [numpoints repmat(numfeat+1,1,order)];
U = cpd_rnd(size_tens(2:end),rank);
W = cpdgen(U);

for ii = 1:numpoints  % Y output. Created as mode-n prod. with tensor
    Un = Umat2(X(ii,:),order);
    Y(ii) = tmprod(W,Un,(1:order));
end
Y = Y';

%% Create initial U0
tic % time
perturbation = cpd_rnd(size_tens(2:end),rank);       % true solution + noise
for ii=1:length(U)
    U0{ii} = U{ii} + factor*perturbation{ii};
end
% U0 = cpd_rnd(size_tens(2:end),rank);               % totally random
% U0 = U;                                            % exactly same initial point

% Algebraic solution

%% Optimization LS-CPD
A = coeffMatrix(numfeat, order, X); % build A coefficient matrix
b = Y;

% Solver options (default algorithm is nls_gndl
optimizer = 'ls-cpd/nls_gndl';
options.Display = true;
options.TolFun = eps^2;
options.TolX = eps; % Caution, can this be smaller than eps?
options.MaxIter = 2000; % default 200
options.TolAbs = eps; % ??
options.CGMaxIter = 120;

% compute solution
[Ures,output] = lscpd_nls(A,b,U0,options);

Wres = cpdgen(Ures);
time = toc;

%% Tests
% Train set
Err = frob(ful(W)-ful(Wres))/frob(ful(W));
disp(['Relative error (tensor, frobenius norm): ',num2str(Err)])

%% Log file
if exist('log2.txt', 'file') ~= 2 % when file does not exist
    fileID = fopen('log2.txt','w');
    formatSpec = ' Rel. error || Time (s) || Iterations || Stop info || Tensor order || Dimensions || Rank || Nº datapoints || Optimizer ||';
    fprintf(fileID,formatSpec);
    fclose(fileID);
    
    dimformat = string('%dx');
    for ii=1:length(size(Wres))-1
        dimformat = strcat(dimformat,'%dx');
    end
    
    fileID = fopen('log2.txt','a+');
    formatSpec = strcat('\n %4.3e || %4.2f || %d || %d || %d || ',dimformat, ' || %d || %d || %10s');
    fprintf(fileID,formatSpec,Err,time,output.iterations,output.info,order,size(Wres),rank,numpoints,optimizer);
    fclose(fileID);
    
elseif exist('log2.txt', 'file') == 2 % when file exists
    fileID = fopen('log2.txt','a+');
    
    % if I still want to write header when file exists...
    formatSpec = '\n Rel. error || Time (s) || Iterations|| Stop info || Tensor order || Dimensions || Rank || Nº Datapoints || Optimizer';
    fprintf(fileID,formatSpec);
    fclose(fileID);
    
    dimformat = string('%dx');
    for ii=1:length(size(Wres))-1
        dimformat = strcat(dimformat,'%dx');
    end
    
    fileID = fopen('log2.txt','a+');
    formatSpec = strcat('\n %4.3e || %4.2f || %d || %d || %d || ',dimformat, ' || %d || %d || %10s');
    fprintf(fileID,formatSpec,Err,time,output.iterations,output.info,order,size(Wres),rank,numpoints,optimizer);
    fclose(fileID);
    
else % any other case
    disp('*** Error in writing file ***')
end
