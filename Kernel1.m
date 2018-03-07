classdef Kernel1 < TensorOptimizationKernel

properties
    % things you want to store, e.g., the data and the labels
    x
    y
    lambda
    numfeat
    polyOrder
    combMat
end

methods

function this = Kernel1(x, y, lambda, numfeat, polyOrder) % constructor
    this.x = x;
    this.y = y;
    this.lambda = lambda;
    this.numfeat = numfeat;
    this.polyOrder = polyOrder;
end

function fval = objfun(this,z) % objective function
    X = this.x;
    Y = this.y;
    for ii=1:length(X) % for all datapoints
        U = Umat(X(ii,:),this.polyOrder);
        Yest(ii) = tmprod(z,U,(1:this.numfeat));  % mode-n tensor-matrix product
    end
    Yest = Yest';
%     fval = sum((Y-Yest).^2) + (this.lambda/2)*(frob(z)).^2; % with regularization
    fval = (1/2)*sum((Y-Yest).^2); % no regularization
end 

function grad = grad(this,z) % column vector with the gradient
    %% analytic
    X = this.x;
    Y = this.y;
    for jj=1:numel(z)
        prod1=zeros(100,1);
        prod2=zeros(100,1);
        [i1,i2,i3,i4,i5] = ind2sub(size(z),jj); % convert loop variable into 5-D subscripts
        expo = [i1,i2,i3,i4,i5]-1; % the subscripts - 1 become the exponents of the variables, present in the gradient
        for ii=1:length(X)
            U = Umat(X(ii,:),this.polyOrder);
            prod1(ii) = -(Y(ii) - tmprod(z,U,(1:this.numfeat)))*prod(X(ii,:).^expo); % derivative 
        end
        grad(jj) = sum(prod1); % derivative
    end

    %% numerical
%     grad = TensorOptimizationKernel.serialize(deriv(@this.objfun, z, this.objfun(z), 'gradient'));
end

function isvalid = validate(this, z)
    isvalid = true; % you can ignore this for now
end

function initialize(this, z) % initialize some things, e.g., cached variables.
    
end 

end

end