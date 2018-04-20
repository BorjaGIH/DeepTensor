clearvars; close all;
rng(1);
% https://github.com/BorjaGIH/DeepTensor
% LS_CPD branch

%% Create feature dataset and output
numfeat = 4;             % Number of features. numfeat+1 is the dimension(s) of the tensor
numpoints = 150;         % Number of datapoints (each datapoint has numfeat values)
order = 3;               % Order of the tensor. "order" is degree of the polynomial that tensor product achieves
X = randi(2,numpoints,numfeat); % Input data, numpoints x numfeat matrix
X = [ones(numpoints,1) X]; % Add bias term
rank = 3;                % Rank of the tensor, for constraint/efficient representation
factor = 4;              % factor to multiply tensor, for "good" initial point (temporary)

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
% U0 = cpd_rnd(size_tens(2:end),rank); % totally random
U0 = U;
% U0 = ful(U) + 0.00000000001*ful(cpd_rnd(size_tens(2:end),rank)); % true ground + noise. careful, how does the rank affect?
% [U0,~] = cpd(U0,rank);

%% Optimization LS-CPD
optimizer = 'ls-cpd/nls_gndl';
A = coeffMatrix(numfeat, order, X); % build A coefficient matrix
b = Y;

% Solver options (default algorithm is nls_gndl
options.Display = true;
options.TolFun = eps^2;
options.TolX = eps; % Caution, can this be smaller than eps?
options.MaxIter = 300; % default 200
options.TolAbs = eps; % ??
% options.CGMaxIter = prod(size_tens(2:end));

% compute solution
[Ures,output] = lscpd_nls(A,b,U0,options);

Wres = cpdgen(Ures);
time = toc;

%% Tests
% Train set
Err = frob(ful(W)-ful(Wres))/frob(ful(W));
disp(['Relative error (tensor, frobenius norm): ',num2str(Err)])

%% Log file
% if exist('log.txt', 'file') ~= 2 % when file does not exist
%     fileID = fopen('log.txt','w');
%     formatSpec = ' Rel. error || Time (s) || Iterations || Stop info || Tensor order || Dimensions || Rank || Nonlin. f || Optimizer ||';
%     fprintf(fileID,formatSpec);
%     fclose(fileID);
%     
%     dimformat = string('%dx');
%     for ii=1:length(size(Wres))-1
%         dimformat = strcat(dimformat,'%dx');
%     end
%     
%     fileID = fopen('log.txt','a+');
%     formatSpec = strcat('\n %4.2f || %4.2f || %d || %d || %d || ',dimformat, ' || %d || %d || %10s');
%     fprintf(fileID,formatSpec,Err,time,output.iterations,output.info,order,size(Wres),rank,nonlin,optimizer);
%     fclose(fileID);
%     
% elseif exist('log.txt', 'file') == 2 % when file exists
%     fileID = fopen('log.txt','a+');
%     
%     % if I still want to write header when file exists...
%     formatSpec = '\n Rel. error || Time (s) || Iterations|| Stop info || Tensor order || Dimensions || Rank || Nonlin. f || Optimizer';
%     fprintf(fileID,formatSpec);
%     fclose(fileID);
%     
%     dimformat = string('%dx');
%     for ii=1:length(size(Wres))-1
%         dimformat = strcat(dimformat,'%dx');
%     end
%     
%     fileID = fopen('log.txt','a+');
%     formatSpec = strcat('\n %4.2f || %4.2f || %d || %d || %d || ',dimformat, ' || %d || %d || %10s');
%     fprintf(fileID,formatSpec,Err,time,output.iterations,output.info,order,size(Wres),rank,nonlin,optimizer);
%     fclose(fileID);
%     
% else % any other case
%     disp('*** Error in writing file ***')
% end
