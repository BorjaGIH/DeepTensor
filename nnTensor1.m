clearvars; close all;
rng(1);
% https://github.com/BorjaGIH/DeepTensor
% PD_constraint_dataTensor branch

%% Create feature dataset and multivariate output
numfeat = 4;             % Number of features. numfeat+1 is the dimension(s) of the tensor
numpoints = 150;         % Number of datapoints (each datapoint has numfeat values)
order = 3;               % Order of the tensor. "order" is degree of the polynomial that tensor product achieves
X = 4*rand(numpoints,numfeat); % Input data, numpoints x numfeat matrix
X = [ones(numpoints,1) X]; % Add bias term
rank = 3;                % Rank of the tensor, for constraint/efficient representation
factor = 1e-4;           % factor to multiply tensor, for "good" initial point (temporary). SNR = abs(exponent)

size_tens = [numpoints repmat(numfeat+1,1,order)];
U = cpd_rnd(size_tens(2:end),rank); % "true" tensor
W = cpdgen(U);

for ii = 1:numpoints  % Y output. Created as mode-n prod. with tensor
    Un = Umat2(X(ii,:),order);
    Y(ii) = tmprod(W,Un,(1:order));
end
Y = Y';

%% Create initial U0
tic % time
perturbation = cpd_rnd(size_tens(2:end),rank);  % true solution + noise
for ii=1:length(U)
    U0{ii} = U{ii} + factor*perturbation{ii};
end
% U0 = cpd_rnd(size_tens(2:end),rank); % totally random
% U0 = U; % exactly same initial point

%% Optimization using kernel

kernel = Kernel1(X,Y,numfeat,order,rank); % create kernel
kernel.initialize(U0); % z0 is the initial guess for the variables, e.g., z0 = W0. lambda is the reg. parameter

%%%%%%%% Notes %%%%%%%%
% Rank of the reconstructed solution tensor is 1, instead of "rank"
% Check if this is normal, seeing that the optimization variables are the
% factor matrices.

% optimization process options
options.TolFun = eps; 
options.MaxIter = 300;
options.TolX = eps;

optimizer = 'minf_lbfgs'; % this must coincide with the used function
[Ures,output] = minf_lbfgs(@kernel.objfun, @kernel.grad, U0, options); % Minimize
time = toc;

%% Tests
Err = frob(ful(U)-ful(Ures))/frob(U);
disp(['Relative error (tensor) frob norm: ',num2str(Err)])

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
