clearvars; close all; clc;
rng(1);
% https://github.com/BorjaGIH/DeepTensor
% LS_CPD branch

%% Parameters, input and output data
numfeat = 6;                % Number of features. "numfeat" is the dimension(s) of the tensor (it includes the bias term)
numpoints = 100;            % Number of datapoints (each datapoint has numfeat values)
order = 3;                  % Order of the tensor. "order" is also degree of the polynomial that tensor product achieves
rank = 1;                   % Rank of the CPD representation
noiseFlag = 'output';         % either 'output', 'tensor', 'both' or 'none' depending on where noise is
factorY = 1e-2;              
factorT = 1e-10;
factor0 = 1e-1;

% Generate data and tensors
X = 4*rand(numpoints,numfeat-1);            % X input
X = [ones(numpoints,1) X];                  % add bias term
size_tens = [numpoints repmat(numfeat,1,order)];
Utrue = cpd_rnd(size_tens(2:end),rank);     % "true" tensor

% good method
% for ii = 1:numpoints
%     Y2(ii) = mtkronprod(W,U,(1:order));
% end
% Y2 = Y2';

% add noise where appropriate
switch(noiseFlag)
    case('output')
        W = cpdgen(Utrue);
        for ii = 1:numpoints
            Un = Umat2(X(ii,:),order);
            Y(ii) = tmprod(W,Un,(1:order));
        end
        noiseY = factorY*rand(size(Y,2),1);
        Y = Y' + noiseY;
        SNRo = 20*log10(norm(Y)/norm(noiseY))
        
    case('tensor')
        pert = cpd_rnd(size_tens(2:end),rank);
        for ii=1:length(Utrue)
            U{ii} = Utrue{ii} + factorT*pert{ii};
        end
        W = cpdgen(U);
        for ii = 1:numpoints
            Un = Umat2(X(ii,:),order);
            Y(ii) = tmprod(W,Un,(1:order));
        end
        Y = Y';
        SNRt = 20*log10(frob(cpdgen(pert))/frob(W))
        
    case('both')
        pert = cpd_rnd(size_tens(2:end),rank);
        for ii=1:length(Utrue)
            U{ii} = Utrue{ii} + factorT*pert{ii};
        end
        W = cpdgen(U);
        for ii = 1:numpoints
            Un = Umat2(X(ii,:),order);
            Y(ii) = tmprod(W,Un,(1:order));
        end
        noiseY = factorY*rand(size(Y,2),1);
        Y = Y' + noiseY;
        SNRo = 20*log10(norm(Y)/norm(noiseY))
        SNRt = 20*log10(frob(cpdgen(pert))/frob(W))
        
    case('none')
        W = cpdgen(Utrue);
        for ii = 1:numpoints
            Un = Umat2(X(ii,:),order);
            Y(ii) = tmprod(W,Un,(1:order));
        end
        Y = Y';
    otherwise
        disp('*** Error in variable "noise" ***')
end


%% Initial value U0
tic                                         % time
pert = cpd_rnd(size_tens(2:end),rank);
for ii=1:length(Utrue)
    U0{ii} = Utrue{ii} + factor0*pert{ii}; % true solution + noise
end
U0 = cpd_rnd(size_tens(2:end),rank);        % totally random
% U0 = U;                                   % exactly same initial point

%% Optimization: LS-CPD
A = coeffMatrix(numfeat, order, X);         % build A coefficient matrix
b = Y;

b2 = A*tens2vec(cpdgen(Utrue),1);

check = b-b2

% Solver options (default algorithm is nls_gndl
optimizer = 'ls-cpd/nls_gndl';
options.Display = true;
options.TolFun = eps^2;
options.TolX = eps;                         % Caution, can this be smaller than eps?
options.MaxIter = 2000;                     % default 200
options.TolAbs = eps;                       % ??
options.CGMaxIter = 10;

% compute solution
[Ures,output] = lscpd_nls(A,b,U0,options);

time = toc;

%% Tests
% Train set
Err = frob(cpdgen(Utrue)-cpdgen(Ures))/frob(cpdgen(Utrue));
disp(['Relative error (tensor, frobenius norm): ',num2str(Err)])

%% Log file
if exist('log2.txt', 'file') ~= 2 % when file does not exist
    fileID = fopen('log2.txt','w');
    formatSpec = ' Rel. error || Time (s) || Iterations || Stop info || Tensor order || Dimensions || Rank || Nº datapoints || Optimizer ||';
    fprintf(fileID,formatSpec);
    fclose(fileID);
    
    dimformat = string('%dx');
    for ii=1:length(size_tens)-2
        dimformat = strcat(dimformat,'%dx');
    end
    
    fileID = fopen('log2.txt','a+');
    formatSpec = strcat('\n %4.3e || %4.2f || %d || %d || %d || ',dimformat, ' || %d || %d || %10s');
    fprintf(fileID,formatSpec,Err,time,output.iterations,output.info,order,size_tens(2:end),rank,numpoints,optimizer);
    fclose(fileID);
    
elseif exist('log2.txt', 'file') == 2 % when file exists
    fileID = fopen('log2.txt','a+');
    
    % if I still want to write header when file exists...
%     formatSpec = '\n Rel. error || Time (s) || Iterations|| Stop info || Tensor order || Dimensions || Rank || Nº Datapoints || Optimizer';
%     fprintf(fileID,formatSpec);
    
    dimformat = string('%dx');
    for ii=1:length(size_tens)-2
        dimformat = strcat(dimformat,'%dx');
    end
    
    formatSpec = strcat('\n %4.3e || %4.2f || %d || %d || %d || ',dimformat, ' || %d || %d || %10s');
    fprintf(fileID,formatSpec,Err,time,output.iterations,output.info,order,size_tens(2:end),rank,numpoints,optimizer);
    fclose(fileID);
    
else % any other case
    disp('*** Error in writing file ***')
end
