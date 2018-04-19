clearvars; close all;
rng(1);
% https://github.com/BorjaGIH/DeepTensor
% PD_constraint_dataTensor branch

%% Create feature dataset and multivariate output
numfeat = 4;             % Number of features. numfeat+1 is the dimension(s) of the tensor
numpoints = 150;         % Number of datapoints (each datapoint has numfeat values)
order = 3;               % Order of the tensor. "order" is degree of the polynomial that tensor product achieves
lambda = 1e8;            % Regularization parameter
X = randi(10,numpoints,numfeat); % Input data, numpoints x numfeat matrix
rank = 3;                % rank of the tensor, for constraint/efficient representation
options.MaxIter = 300;   % optimization iterations
nonlin = true;           % learned function is nonlinear

% Y output. Created as mode-n prod. with tensor
initvec = repmat(numfeat+1,1,order);
W = rand(initvec); % Dense random "TRUE" tensor

for ii = 1:numpoints
    U = Umat2(X(ii,:),order);
    Ys(ii) = tmprod(W,U,(1:order));
end
Ys = Ys';

% Divide dataset for train and test
sizetr = 0.7;
sizete = 0.3;

Xtr = X(1:sizetr*numpoints,:);
Ystr = Ys(1:sizetr*numpoints); % MISO

Xte = X(1:sizete*numpoints,:);
Yste = Ys(1:sizete*numpoints);

%% Create initial random tensor/find "good" initial point
tic % time
W0 = rand(initvec); % Dense random initial tensor
% To do: find good initial point

%% Optimization using kernel
[U0,~] = cpd(W0,rank); % factorize and ensure we take a rank smaller than that estimated by rankest
% R = rankest(W0);
% if rank>R
%     rank = R;
%     disp(['Inputed rank is bigger than estimated by rankest, so it has been changed '])
% end

kernel = Kernel1(Xtr,Ystr,lambda,numfeat,order,rank); % create kernel
kernel.initialize(U0); % z0 is the initial guess for the variables, e.g., z0 = W0. lambda is the reg. parameter

%%%%%%%% Notes %%%%%%%%
% Rank of the reconstructed solution tensor is 1, instead of "rank"
% Check if this is normal, seeing that the optimization variables are the
% factor matrices.

% some optimization process options
options.TolFun = 1e-20; 
% options.Display = 10;
% options.TolX = 1e-20;

optimizer = 'minf_lbfgsdl'; % this must coincide with the used function
[Ures,output] = minf_lbfgsdl(@kernel.objfun, @kernel.grad, U0, options); % Minimize
Wres = cpdgen(Ures); % reconstruct tensor from factors
time = toc;

%% Tests
Err = (frob(W)-frob(Wres))/frob(W)
disp(['Relative error (tensor) frob norm: ',Err])

%% Log file
if exist('log.txt', 'file') ~= 2 % when file does not exist
    fileID = fopen('log.txt','w');
    formatSpec = ' Rel. error || Time (s) || Iterations || Stop info || Tensor order || Dimensions || Rank || Nonlin. f || Optimizer ||';
    fprintf(fileID,formatSpec);
    fclose(fileID);
    
    dimformat = string('%dx');
    for ii=1:length(size(Wres))-1
        dimformat = strcat(dimformat,'%dx');
    end
    
    fileID = fopen('log.txt','a+');
    formatSpec = strcat('\n %4.2f || %4.2f || %d || %d || %d || ',dimformat, ' || %d || %d || %10s');
    fprintf(fileID,formatSpec,Err,time,output.iterations,output.info,order,size(Wres),rank,nonlin,optimizer);
    fclose(fileID);
    
elseif exist('log.txt', 'file') == 2 % when file exists
    fileID = fopen('log.txt','a+');
    
    % if I still want to write header when file exists...
    formatSpec = '\n Rel. error || Time (s) || Iterations|| Stop info || Tensor order || Dimensions || Rank || Nonlin. f || Optimizer';
    fprintf(fileID,formatSpec);
    fclose(fileID);
    
    dimformat = string('%dx');
    for ii=1:length(size(Wres))-1
        dimformat = strcat(dimformat,'%dx');
    end
    
    fileID = fopen('log.txt','a+');
    formatSpec = strcat('\n %4.2f || %4.2f || %d || %d || %d || ',dimformat, ' || %d || %d || %10s');
    fprintf(fileID,formatSpec,Err,time,output.iterations,output.info,order,size(Wres),rank,nonlin,optimizer);
    fclose(fileID);
    
else % any other case
    disp('*** Error in writing file ***')
end
