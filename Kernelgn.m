classdef Kernelgn < TensorOptimizationKernel

properties
    x       % input data
    y       % label or output data
    lambda  % regularization parameter (when needed)
    numfeat % number of features 
    N       % order of the tensor
    R       % rank of the tensor for decomposition (constraint/efficient representation)
    resid   % residual, r(z), that is used in J, JHJ, etc.
    jacobian    % jacobian, used in JHJx etc.
    JHJinv  % preconditioner
end

methods

function this = Kernelgn(x, y, numfeat, N, R, resid, jacobian, JHJinv) % constructor
    this.x = x;
    this.y = y;
    this.numfeat = numfeat;
    this.N = N;
    this.R = R;
    this.resid = resid;
    this.jacobian = jacobian;
    this.JHJinv = JHJinv;
end

function fval = objfun(this,z) % objective function
    X = this.x;
    Y = this.y;
    mult2 = 1;
    for n=1:this.N
        mult2 = mult2.*(X*z{n});
    end
    r = (mult2*ones(this.R,1));
    this.resid = r-Y;
    fval = 0.5*(this.resid'*this.resid);
end 

function grad = grad(this,z) % column vector with the gradient
    X = this.x;
    E = this.resid;
    grad = zeros(this.N*this.R*this.numfeat,1);
    gradind=1;

    for n = 1:this.N
        for ii = 1:this.R*this.numfeat
            zaux = z;
            zaux{n} = zeros(size(zaux{n}));
            zaux{n}(ii) = 1;
            tmp = cellfun(@(u) X*u, zaux, 'UniformOutput', false);
            mult = 1;
            for k = 1:this.N
                mult = mult.*tmp{k};
            end
            this.jacobian{n}(:,ii) = mult*ones(this.R,1); % make it property
            grad(gradind) = this.jacobian{n}(:,ii)'*E;
            gradind = gradind+1;
        end
        this.JHJinv{n} = pinv(conj(this.jacobian{n}.')*this.jacobian{n});
    end
        
    %% Assert
%     tol = 1e-5;
% 
%     % target
%     grad1 = deriv(@this.objfun, z, this.objfun(z), 'gradient');
%     grad1 = TensorOptimizationKernel.serialize(grad1);
%     % computed
%     grad2 = gradn;
%     
%     % Check if correct
%     err = frob(grad1-grad2);
%     relerr = frob(grad1-grad2)/frob(grad1);
% 
% %     assert(relerr <= tol);
%     if ~(relerr <= tol)
%         disp(['Rel. err: ',num2str(relerr)]);
%         disp(['Err: ',num2str(err)]);
%         return
%     end
end

function y = JHJx(this,z,y)
%     yold = y; % for the numerical check
    
    % Tensor-vector product LS-CPD Gramian vector product. Implement fast via blocks
        offset = cellfun(@(u) numel(u),z(:));
        offset = cumsum([1; kron(offset,1)]);
        J  = this.jacobian;
        Jx = zeros(size(this.x,1),1);
        
        for n = 1:this.N
            Jx = Jx + (J{n})*y(offset(n):offset(n+1)-1);
        end
        for n = 1:this.N
            y(offset(n):offset(n+1)-1) = (J{n})'*Jx;
        end
        
    %% Assert
%     model = @(Z) residFun(this,Z);
%     fun = 1;
%     elementwise = isnumeric(fun) || nargin(fun) == 2; 
%     tol = 1e-6;
%     
%     % target
%     J1 = deriv(model, z, [], 'Jacobian'); % error, throws same response if 4th parameter is "gradient"
%     M  = reshape(model(z), [], 1);
%     if isnumeric(fun)
%         D1 = fun;
%         if isvector(D1), D1 = diag(D1); end
%     else
%         D1 = TensorOptimizationKernel.numericHessianFD(fun, M, elementwise);
%     end
%     y1 = J1'*(D1*(J1*yold));
%     
%     % computed
%     y2 = y;
%     
%     % Check if correct
%     relerr = frob(y1-y2)/frob(y1);
%     
% %     assert(relerr <= tol);
%     if ~(relerr <= tol)
%         disp(['Rel. err: ',num2str(relerr)]);
%     end
end

function y = M_jacobi(this,z,q)
    % Solve M*y = q, where M is an approximation for JHJ.
    y = zeros(size(q));
    offset = cellfun(@(u) numel(u),z(:));
    offset = cumsum([1; kron(offset,1)]);
    for n = 1:this.N
        idx1 = offset(n):offset(n+1)-1;
        y(idx1) = this.JHJinv{n}*q(idx1);
    end
    
end

function res = residFun(this,z)
    X = this.x;
    Y = this.y;    
    mult2 = 1;
    for n=1:this.N
        mult2 = mult2.*(X*z{n});
    end
    r = (mult2*ones(this.R,1));
    res = r-Y;
end

function isvalid = validate(this, z)
    isvalid = true; % you can ignore this for now
end

function initialize(this, z) % initialize some things, e.g., cached variables.
    
end 

end

end