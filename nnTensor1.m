clearvars; close all; clc;
% rng(1);
% https://github.com/BorjaGIH/DeepTensor
% LS_CPD branch

%% Parameters
numfeat = 10;                    % Number of features. "numfeat" is the dimension(s) of the tensor (it includes the bias term)
order = 4;                      % Order of the tensor. "order" is also degree of the polynomial that tensor product achieves
R = 2;                          % Rank of the CPD representation
noiseFlag = 'none';             % either 'output', 'tensor', 'both' or 'none' depending on where noise is
factorY = 1e-2;                % factor for the noise in output
factorT = 1e-2;                % factor for the noise in tensor
factor0 = 2;                    % factor for the initial value
facX = 3;                       % factor for the inputs
Mmin = (numfeat*order-order+1)*R+1; % Lemma 1, datapoints (M) must be bigger than or equal to: M>=(I1+I2...+In-N+1)R+1
numpoints = 15;                  % Number of datapoints (each datapoint has numfeat values)
generator = 'function';            % either 'tensor' or 'function'
ratioTr = 0.7;                  % Fraction of the datapoints used for train
ratioTe = 1-ratioTr;            % Fraction of the datapoints used for test

optimizer = 'ls-cpd/nls_gndl';  % optimizer and optimizer options
options.Display = 10;
options.TolFun = eps^2;
options.TolX = eps;
options.MaxIter = 100;
options.TolAbs = eps;
options.CGMaxIter = 60;

%% Generate data and tensors
X = facX*rand(numpoints,numfeat-1);            % X input
X = [ones(numpoints,1) X];                  % add bias term
size_tens = repmat(numfeat,1,order);
Utrue = cpd_rnd(size_tens(:),R);         % "true" tensor
 
if strcmp(generator,'tensor')
    % add noise where appropriate
    switch(noiseFlag)
        case('output')
            W = cpdgen(Utrue);
            Un = repmat({X'},1,order);
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
            Un = repmat({X'},1,order);
            Y = mtkrprod(W,Un,0)';
            SNRt = 20*log10(frob(cpdgen(pert))/frob(W))
            
        case('both')
            pert = cpd_rnd(size_tens(:),R);
            for ii=1:length(Utrue)
                U{ii} = Utrue{ii} + factorT*pert{ii};
            end
            W = cpdgen(U);
            Un = repmat({X'},1,order);
            Y = mtkrprod(W,Un,0)';
            noiseY = factorY*rand(size(Y,2),1);
            Y = Y + noiseY;
            SNRo = 20*log10(norm(Y)/norm(noiseY))
            SNRt = 20*log10(frob(cpdgen(pert))/frob(W))
            
        case('none')
            W = cpdgen(Utrue);
            Un = repmat({X'},1,order);
            Y = mtkrprod(W,Un,0)';
        otherwise
            disp('*** Error in "noise" variable ***')
    end
 
elseif strcmp(generator,'function')
    Y = genfun(X,order,numfeat);
    Y = Y';
end

X = X(1:floor(ratioTr*length(X)),:);
Xte = X(floor(ratioTr*length(X))+1:end,:);
Y = Y(1:floor(ratioTr*length(Y)));
Yte = Y(floor(ratioTr*length(Y))+1:end);

%% Initial value, A and b
tic  % start time

% Initial value
U0 = cpd_rnd(size_tens(:),R);            % random

% Compute A and b
A = kr(repmat({X'},1,order))';
b = Y;
disp(['Size of A:',num2str(size(A))])
disp(['rank of A:',num2str(rank(A))])

%% Optimization LS-CPD
[Ures,output] = lscpd_nls(A,b,U0,options);
time = toc;   % end time

%% Test
if strcmp(generator,'tensor')
    % Train set
    % Plot of the error
    err = (sqrt(output.fval(2:end)*2))/norm(Y);
    semilogy(err); xlabel('Iteration'); ylabel('error');
    
    Wres = cpdgen(Ures);
    Un = repmat({X'},1,order);
    Yres = mtkrprod(Wres,Un,0)';
    
    ErrT = frob(W-Wres)/frob(W);        % Error in tensor
    ErrY = norm(Y-Yres)/norm(Y);
    disp(['Relative error of tensor, frobenius norm: ',num2str(ErrT)])
    disp(['Relative error of Yest train, 2-norm: ',num2str(ErrY)])
    
    % Test set
    Un = repmat({Xte'},1,order);
    YresTe = mtkrprod(Wres,Un,0)';
    ErrYte = norm(Yte-YresTe)/norm(Yte);
    disp(['Relative error of Yest test, 2-norm: ',num2str(ErrYte)])

elseif strcmp(generator,'function')
    % Train set
    % Plot of the error
    err = (sqrt(output.fval(2:end)*2))/norm(Y);
    semilogy(err); xlabel('Iteration'); ylabel('error');
    
    Un = repmat({X'},1,order);
    Wres = cpdgen(Ures);
    Yres = mtkrprod(Wres,Un,0)';
    
    ErrY = norm(Y-Yres)/norm(Y);
    disp(['Relative error of Yest train, 2-norm: ',num2str(ErrY)])
    
    % Test set
    Un = repmat({Xte'},1,order);
    YresTe = mtkrprod(Wres,Un,0)';

    ErrYte = norm(Yte-YresTe)/norm(Yte);
    disp(['Relative error of Yest test, 2-norm: ',num2str(ErrYte)])
end

%% Log file
% if exist('log2.txt', 'file') ~= 2 % when file does not exist
%     fileID = fopen('log2.txt','w');
%     formatSpec = ' Rel. error || Time (s) || Iterations || Stop info || Tensor order || Dimensions || Rank || Nº datapoints || Optimizer ||';
%     fprintf(fileID,formatSpec);
%     fclose(fileID);
%     
%     dimformat = string('%dx');
%     for ii=1:length(size_tens)-1
%         dimformat = strcat(dimformat,'%dx');
%     end
%     
%     fileID = fopen('log2.txt','a+');
%     formatSpec = strcat('\n %4.3e || %4.2f || %d || %d || %d || ',dimformat, ' || %d || %d || %10s');
%     fprintf(fileID,formatSpec,Err,time,output.iterations,output.info,order,size_tens(:),rank,numpoints,optimizer);
%     fclose(fileID);
%     
% elseif exist('log2.txt', 'file') == 2 % when file exists
%     fileID = fopen('log2.txt','a+');
%     
%     % if I still want to write header when file exists...
% %     formatSpec = '\n Rel. error || Time (s) || Iterations|| Stop info || Tensor order || Dimensions || Rank || Nº Datapoints || Optimizer';
% %     fprintf(fileID,formatSpec);
%     
%     dimformat = string('%dx');
%     for ii=1:length(size_tens)-1
%         dimformat = strcat(dimformat,'%dx');
%     end
%     
%     formatSpec = strcat('\n %4.3e || %4.2f || %d || %d || %d || ',dimformat, ' || %d || %d || %10s');
%     fprintf(fileID,formatSpec,Err,time,output.iterations,output.info,order,size_tens(:),rank,numpoints,optimizer);
%     fclose(fileID);
%     
% else % any other case
%     disp('*** Error in writing file ***')
% end
