clearvars; clc;
rng(1);
% https://github.com/BorjaGIH/DeepTensor
% % PD_constraint_dataTensor branch

%% Parameters
numfeat = 4;                    % Number of features. "numfeat" is the dimension(s) of the tensor (it includes the bias term)
N = 4;                      % Order of the tensor. "order" is also degree of the polynomial that tensor product achieves
R = 2;                          % Rank of the CPD representation
M = 200;                         % Number of datapoints (each datapoint has numfeat values)
generator = 'tensor';           % either 'tensor' or 'function'
ratioTr = 0.7;                  % fraction of datapoints to use for train
ratioTe = 1 - ratioTr;          % fraction of datapoints to use for test
noiseFlag = 'none';             % either 'output', 'tensor', 'both' or 'none' depending on where noise is
factorY = 1e0;                 % factor for the noise in output
factorT = 1e-2;                 % factor for the noise in tensor
factor0 = 2;                    % factor for the initial value
facX = 1;                       % factor for the random datapoints

optimizer = 'minf_lfbgs';  % optimizer and optimizer options
options.Display = true;
options.TolFun = eps;
options.TolX = eps;
options.MaxIter = 3000;
options.TolAbs = eps;

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

%% Initial value
% Initial value
U0 = cpd_rnd(size_tens(:),R);            % random

%% Optimization using kernel
tic  % start time
% Kernels
% kernel1 = Kernelbfgs(Xtr,Ytr,numfeat,N,R); % create kernel
% kernel1.initialize(U0); % z0 is the initial guess for the variables, e.g., z0 = U0
% [Uest,output] = minf_lbfgs(@kernel1.objfun, @kernel1.grad, U0, options); % Minimize

kernel2 = Kernelgn(Xtr,Ytr,numfeat,N,R,[],[]); % create kernel
kernel2.initialize(U0); % z0 is the initial guess for the variables, e.g., z0 = U0
dF.JHJx = @kernel2.JHJx;
dF.JHF = @kernel2.grad;
[Uest,output] = nls_gndl(@kernel2.objfun, dF, U0);


% Functions
% [Uest,output] = nls_gndl(@objfun, 'Jacobian', U0);

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

%% Functions
% function fval = objfun(z) % objective function
%     X = Xtr;
%     Y = Ytr;
%     npoints = size(X,1);
%     Yest = zeros(npoints,1);
%     
%     % Construct small matrix stacking rows
%     for jj=1:R
%         b1 = [z{1}(:,jj)';zeros(this.N-1,this.numfeat)]; % block
%         for ii=2:this.N
%             b1 = [b1, [zeros(ii-1,this.numfeat);z{ii}(:,jj)';zeros(this.N-ii,this.numfeat)]]; % jj is the rank
%         end
%         if jj==1
%             m1 = b1; % matrix (M1)
%         else
%             m1 = [m1;b1];
%         end
%     end
%     
%     Xii = repmat(X,1,this.N);
%     for ii=1:npoints % loop through all datapoints
%         xii = Xii(ii,:)';
%         tmp = m1*xii;
%         res = reshape(tmp,this.N,this.R);
%         Yest(ii) = sum(prod(res,1));
%     end
%     
%     this.resid = Yest-Y;
%     fval = 0.5*(this.resid'*this.resid);
% end 

% function grad = grad(this,z) % column vector with the gradient
% %% analytic
%     X = this.x;
%     Y = this.y;
%     npoints = size(X,1);
%     gradTmp = zeros(this.N*this.R*this.numfeat,1);
%     
%     indx = repmat(1:this.numfeat,1,this.R*this.N);
%     tmp = sort(repmat(1:this.R,1,this.N));
%     rvec = repmat(tmp,1,this.N);
%     nvec = sort(indx);
%         
%     for ii=1:npoints % loop through all datapoints
%         
%         der = zeros(length(indx),1);
%         for jj=1:length(indx)
%             k = indx(jj);
%             n = nvec(jj);
%             r = rvec(jj);
%             
%             % Direct calculation
%             tmp2 = zeros(1,this.N);
%             ztmp = z;
%             for l=1:this.N
%                 ztmp{n}(:,r) = 0;
%                 ztmp{n}(k,r) = 1;
%                 tmp2(l) = dot(ztmp{l}(:,r),X(ii,:));
%             end
%             der(jj) = prod(tmp2);
%         end  
%         gradTmp = gradTmp + (this.resid(ii)).*der; % Can be written as multilinear op. (kron)
%     end
%     grad = gradTmp;
%     
% end
