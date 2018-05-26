clearvars; clc;
% rng(90);
% https://github.com/BorjaGIH/DeepTensor
% % PD_constraint_dataTensor branch

%% Parameters
numfeat = 4;                    % Number of features. "numfeat" is the dimension(s) of the tensor (it includes the bias term)
N = 4;                      % Order of the tensor. "order" is also degree of the polynomial that tensor product achieves
R = 2;                          % Rank of the CPD representation
M = 200;                         % Number of datapoints (each datapoint has numfeat values)
generator = 'function';           % either 'tensor' or 'function'
ratioTr = 0.7;                  % fraction of datapoints to use for train
ratioTe = 1 - ratioTr;          % fraction of datapoints to use for test
noiseFlag = 'none';             % either 'output', 'tensor', 'both' or 'none' depending on where noise is
factorY = 1e0;                 % factor for the noise in output
factorT = 1e-2;                 % factor for the noise in tensor
factor0 = 2;                    % factor for the initial value
facX = 1;                       % factor for the random datapoints

optimizer = 'minf_lfbgs';  % optimizer and optimizer options
options.Display = true;
options.TolFun = eps^2;
options.TolX = eps;
options.MaxIter = 200;
options.TolAbs = eps;
options.CGmaxIter = 15;

%% Generate data and tensors
X = facX*rand(M,numfeat-1);            % X input
X = [ones(M,1) X];                      % add bias term
size_tens = repmat(numfeat,1,N);
Utrue = cpd_rnd(size_tens(:),R);         % "true" tensor

if strcmp(generator,'tensor')
    % add noise where appropriate
    switch(noiseFlag)
        case('output')
            W = cpdgen(Utrue);
            Un = repmat({X'},1,N);
            Y = mtkrprod(W,Un,0)';
            noiseY = factorY*rand(size(Y,2),1);
            Y = Y + noiseY;
            SNRo = 20*log10(norm(Y)/norm(noiseY))
            
        case('tensor')
            pert = cpd_rnd(size_tens(:),R);
            for ii=1:length(Utrue)
                U{ii} = Utrue{ii} + factorT*pert{ii};
            end
            W = cpdgen(U);
            Un = repmat({X'},1,N);
            Y = mtkrprod(W,Un,0)';
            SNRt = 20*log10(frob(cpdgen(pert))/frob(W))
            
        case('both')
            pert = cpd_rnd(size_tens(:),R);
            for ii=1:length(Utrue)
                U{ii} = Utrue{ii} + factorT*pert{ii};
            end
            W = cpdgen(U);
            Un = repmat({X'},1,N);
            Y = mtkrprod(W,Un,0)';
            noiseY = factorY*rand(size(Y,2),1);
            Y = Y + noiseY;
            SNRo = 20*log10(norm(Y)/norm(noiseY))
            SNRt = 20*log10(frob(cpdgen(pert))/frob(W))
            
        case('none')
            W = cpdgen(Utrue);
            Un = repmat({X'},1,N);
            Y = mtkrprod(W,Un,0)';
        otherwise
            disp('*** Error in "noise" variable ***')
    end

elseif strcmp(generator,'function')
    Y = genfun(X,N,numfeat);
    Y = Y';
end

Xtr = X(1:floor(ratioTr*size(X,1)),:);
Xte = X(floor(ratioTr*size(X,1))+1:end,:);
Ytr = Y(1:floor(ratioTr*size(Y,1)));
Yte = Y(floor(ratioTr*size(Y,1))+1:end);

%% Optimization using kernel
tic  % start time

U0 = cpd_rnd(size_tens(:),R);  % random initial value

% kernel1 = Kernelbfgs(Xtr,Ytr,numfeat,N,R); % create kernel
% kernel1.initialize(U0); % z0 is the initial guess for the variables, e.g., z0 = U0
% [Uest,output] = minf_lbfgs(@kernel1.objfun, @kernel1.grad, U0, options); % Minimize

kernel2 = Kernelgn(Xtr,Ytr,numfeat,N,R,[],[],[]); % create kernel
kernel2.initialize(U0); % z0 is the initial guess for the variables, e.g., z0 = U0
dF.JHF = @kernel2.grad;
dF.JHJx = @kernel2.JHJx;
dF.M = @kernel2.M_jacobi;
[Uest,output] = nls_gndl(@kernel2.objfun, dF, U0, options);

time = toc;

%% Tests
if strcmp(generator,'tensor')
    % Train set
    % Plot of the error
    err = (sqrt(output.fval*2))/norm(Ytr);
    semilogy(err); xlabel('Iteration'); ylabel('error');
    
    West = cpdgen(Uest);
    Un = repmat({Xtr'},1,N);
    Yest = mtkrprod(West,Un,0)';
    
    ErrT = frob(W-West)/frob(W);        % Error in tensor
    ErrY = norm(Ytr-Yest)/norm(Ytr);        % Error in output
    disp(['Relative error of tensor, frobenius norm: ',num2str(ErrT)])
    disp(['Relative error of Yest train, 2-norm: ',num2str(ErrY)])
    
    % Test set
    Un = repmat({Xte'},1,N);
    YestTe = mtkrprod(West,Un,0)';
    ErrYte = norm(Yte-YestTe)/norm(Yte);    % error in output
    disp(['Relative error of Yest test, 2-norm: ',num2str(ErrYte)])
    
    disp(['Time: ',num2str(time)])
    
elseif strcmp(generator,'function')
    % Plot of the error
    err = (sqrt(output.fval*2))/norm(Ytr);
    semilogy(err); xlabel('Iteration'); ylabel('error');
    
    West = cpdgen(Uest);
    Un = repmat({Xtr'},1,N);
    Yest = mtkrprod(West,Un,0)';
    
    ErrY = norm(Ytr-Yest)/norm(Ytr);   
    disp(['Relative error of Yest train, 2-norm: ',num2str(ErrY)])
    
    % Test set
    Un = repmat({Xte'},1,N);
    YestTe = mtkrprod(West,Un,0)';
    ErrYte = norm(Yte-YestTe)/norm(Yte);
    disp(['Relative error of Yest test, 2-norm: ',num2str(ErrYte)])
    
    disp(['Time: ',num2str(time)])
end

%% Log file
% if exist('log3.txt', 'file') ~= 2 % when file does not exist
%     fileID = fopen('log3.txt','w');
%     formatSpec = ' Rel. error || Time (s) || Iterations || Stop info || Tensor order || Dimensions || Rank || Nº Datapoints || Optimizer ||';
%     fprintf(fileID,formatSpec);
%     fclose(fileID);
%     
%     dimformat = string('%dx');
%     for ii=1:length(size(Wres))-1
%         dimformat = strcat(dimformat,'%dx');
%     end
%     
%     fileID = fopen('log3.txt','a+');
%     formatSpec = strcat('\n %4.3e || %4.2f || %d || %d || %d || ',dimformat, ' || %d || %d || %10s');
%     fprintf(fileID,formatSpec,Err,time,output.iterations,output.info,order,size(Wres),rank,numpoints,optimizer);
%     fclose(fileID);
%     
% elseif exist('log3.txt', 'file') == 2 % when file exists
%     fileID = fopen('log3.txt','a+');
%     
%     % if I still want to write header when file exists...
% %     formatSpec = '\n Rel. error || Time (s) || Iterations|| Stop info || Tensor order || Dimensions || Rank || Nº Datapoints || Optimizer';
% %     fprintf(fileID,formatSpec);
%     
%     dimformat = string('%dx');
%     for ii=1:length(size(Wres))-1
%         dimformat = strcat(dimformat,'%dx');
%     end
% 
%     formatSpec = strcat('\n %4.3e || %4.2f || %d || %d || %d || ',dimformat, ' || %d || %d || %10s');
%     fprintf(fileID,formatSpec,Err,time,output.iterations,output.info,order,size(Wres),rank,numpoints,optimizer);
%     fclose(fileID);
%     
% else % any other case
%     disp('*** Error in writing file ***')
% end
