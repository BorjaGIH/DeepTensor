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
%     grad = sqrt(2*objfun(this,z))*;

    %% numerical
    grad = TensorOptimizationKernel.serialize(deriv(@this.objfun, z, this.objfun(z), 'gradient'));
end

function isvalid = validate(this, z)
    isvalid = true; % you can ignore this for now
end

function initialize(this, z) % initialize some things, e.g., cached variables.
    
end 

end

end