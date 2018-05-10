classdef Kernel1 < TensorOptimizationKernel

properties
    x       % input data
    y       % label or output data
    lambda  % regularization parameter (when needed)
    numfeat % number of features 
    N       % order of the tensor
    R       % rank of the tensor for decomposition (constraint/efficient representation)
end

methods

function this = Kernel1(x, y, numfeat, N, R) % constructor
    this.x = x;
    this.y = y;
    this.numfeat = numfeat;
    this.N = N;
    this.R = R;
end

function fval = objfun(this,z) % objective function
    X = this.x;
    Y = this.y;
    npoints = length(X);
    Yest = zeros(npoints,1);
    
    % Construct small matrix stacking rows
    for jj=1:this.R
        b1 = [z{1}(:,jj)';zeros(this.N-1,this.numfeat)]; % block
        for ii=2:this.N
            b1 = [b1, [zeros(ii-1,this.numfeat);z{ii}(:,jj)';zeros(this.N-ii,this.numfeat)]]; % jj is the rank
        end
        if jj==1
            m1 = b1; % matrix (M1)
        else
            m1 = [m1;b1];
        end
    end
    
    Xii = repmat(X,1,this.N);
    for ii=1:npoints % loop through all datapoints
        xii = Xii(ii,:)';
        tmp = m1*xii;
        res = reshape(tmp,this.N,this.R);
        Yest(ii) = sum(prod(res,1));
    end
    
    fval = (1/2)*sum((Y-Yest).^2); % no regularization
end 

function grad = grad(this,z) % column vector with the gradient
    %% analytic
%     X = this.x;
%     Y = this.y;
%     npoints = length(X);
%     Yest = zeros(npoints,1);
%     
%     % Construct small matrix stacking rows
%     for jj=1:this.R
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
%     Xjj = repmat(X,1,this.N*this.R);
%     gradTmp = zeros(this.N*this.R*this.numfeat,1);
%     for ii=1:npoints % loop through all datapoints
%         xii = Xii(ii,:)';
%         tmp = m1*xii;
%         res = reshape(tmp,this.N,this.R);
%         Yest(ii) = sum(prod(res,1));
%         
%         rvec = repmat((1 : 1 : this.R),1,this.R);
%         nvec = sort(rvec);
%         reord = (1 : this.N : this.N*(this.R-1)+1);
%         for ii=2:this.R
%             reord = [reord, (ii : this.N : this.N*(this.R-1)+ii)];
%         end
%         
%         for jj=0:this.R*this.N*this.numfeat-1
% %             disp(['jj=',num2str(jj+1)])
%             k = floor(jj/(this.numfeat))+1;
%             n = nvec(k);
% %             disp(['n=',num2str(n)])
%             r = rvec(k);
% %             disp(['r=',num2str(r)])
%             
%             tmp2 = tmp;
%             tmp2(reord(k))=Xjj(ii,jj+1);
%             res = reshape(tmp2,this.N,this.R);
%             tmp3 = prod(res,1);
%             der(jj+1) = tmp3(r);
% %             disp('****')
%         end  
%         gradTmp = gradTmp + (Yest(ii)-Y(ii)).*der';
%     end
%     grad = gradTmp;
    
    %% numerical
    grad = TensorOptimizationKernel.serialize(deriv(@this.objfun, z, this.objfun(z), 'gradient'));
    
%     plot(grad); hold on; plot(grad2); hold off;
%     close;
end

function isvalid = validate(this, z)
    isvalid = true; % you can ignore this for now
end

function initialize(this, z) % initialize some things, e.g., cached variables.
    
end 

end

end