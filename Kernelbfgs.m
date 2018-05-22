classdef Kernelbfgs < TensorOptimizationKernel

properties
    x       % input data
    y       % label or output data
    lambda  % regularization parameter (when needed)
    numfeat % number of features 
    N       % order of the tensor
    R       % rank of the tensor for decomposition (constraint/efficient representation)
end

methods

function this = Kernelbfgs(x, y, numfeat, N, R) % constructor
    this.x = x;
    this.y = y;
    this.numfeat = numfeat;
    this.N = N;
    this.R = R;
end

function fval = objfun(this,z) % objective function
    X = this.x;
    Y = this.y;
    npoints = size(X,1);
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
    X = this.x;
    Y = this.y;
    npoints = size(X,1);
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
    gradTmp = zeros(this.N*this.R*this.numfeat,1);
    
    indx = repmat(1:this.numfeat,1,this.R*this.N);
    tmp = sort(repmat(1:this.R,1,this.N));
    rvec = repmat(tmp,1,this.N);
    nvec = sort(indx);
        
    for ii=1:npoints % loop through all datapoints
        xii = Xii(ii,:)';
        tmp = m1*xii;
        res = reshape(tmp,this.N,this.R);
        Yest(ii) = sum(prod(res,1));
        
        der = zeros(length(indx),1);
        for jj=1:length(indx)
            k = indx(jj);
            n = nvec(jj);
            r = rvec(jj);
            
            % Direct calculation
            tmp2 = zeros(1,this.N);
            ztmp = z;
            for l=1:this.N
                ztmp{n}(:,r) = 0;
                ztmp{n}(k,r) = 1;
                tmp2(l) = dot(ztmp{l}(:,r),X(ii,:));
            end
            der(jj) = prod(tmp2);
        end  
        gradTmp = gradTmp + (Yest(ii)-Y(ii)).*der; % Can be written as multilinear op. (kron)
    end
    grad = gradTmp;
    
    %% numerical
%     grad1 = TensorOptimizationKernel.serialize(deriv(@this.objfun, z, this.objfun(z), 'gradient'));
    % include assertion in case gradient fails
    
end

function jhj = JHDJ(this, z)
    %JHDJ Compute an approximation of the Hessian.
    %   JHJ = JHDJ(Z) computes a positive semidefinite approximation of the Hessian
    %   of the objective function using variables Z. This approximation is
    %   usually the Gramian matrix, or its generalization to other
    %   divergences.
    
    
end

function y = JHDJx(this, z, x)
    %JHDJX Compute the approximate Hessian vector product.
    %   Y = JHDJX(Z,X) computes the product of the positive semidefinite
    %   approximation of the Hessian of the objective function constructed
    %   using variables Z, with a vector X. The approximation is usually the
    %   Gramian matrix, or its generalization to other divergences.
    
    
end

function isvalid = validate(this, z)
    isvalid = true; % you can ignore this for now
end

function initialize(this, z) % initialize some things, e.g., cached variables.
    
end 

end

end