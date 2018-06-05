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
    gramian
end

methods

function this = Kernelgn(x, y, numfeat, N, R, resid, jacobian, gramian) % constructor
    this.x = x;
    this.y = y;
    this.numfeat = numfeat;
    this.N = N;
    this.R = R;
    this.resid = resid;
    this.jacobian = jacobian;
    this.gramian = gramian;
end

function fval = objfun(this,z) % objective function
    X = this.x;
    Y = this.y;
    U = z{1};
    S = z{2};
    
    Ux = cellfun(@(u) u'*X',U,'UniformOutput',false);
    Uxf = flip(Ux);
    Yest = S(:)'*kr(Uxf);
    
    this.resid = Yest'-Y;
    fval = 0.5*(this.resid'*this.resid);
end 

function grad = grad(this,z) % column vector with the gradient
    % Tensor-vector product LS-CPD gradient
      X = this.x;
      E = this.resid;
        M = size(E,1);
        grad = zeros(this.N*this.R*this.numfeat+this.R^this.N,1);
        gradind=1;
        U = z{1};
        S = z{2};
        
        % Part of U
        for n = 1:this.N
            for ii = 1:this.R*this.numfeat
                zaux = U;
                zaux{n} = zeros(size(zaux{n}));
                zaux{n}(ii) = 1;
                Ux = cellfun(@(u) u'*X', zaux, 'UniformOutput', false);
                Uxf = flip(Ux);
                this.jacobian{n}(:,ii) = S(:)'*kr(Uxf);
                grad(gradind) = this.jacobian{n}(:,ii)'*E;
                gradind = gradind+1;
            end
%         this.JHJinv{n} = pinv(conj(this.jacobian{n}.')*this.jacobian{n});
        end
        
        % Part of S
        for ii=1:numel(S)
            saux = zeros(numel(S),1);
            saux(ii)= 1;
            Ux = cellfun(@(u) u'*X', U, 'UniformOutput', false);
            Uxf = flip(Ux);
            this.jacobian{this.N+1}(:,ii) = saux(:)'*kr(Uxf);
            grad(gradind) = this.jacobian{this.N+1}(:,ii)'*E;
            gradind = gradind+1;
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
    yold = y;
% 
    J  = this.jacobian;
%     Jx = zeros(size(this.x,1),1);
%     offset = cellfun(@(u) size(u,2),this.jacobian);
%     offset = cumsum([1; kron(offset,1)']);
%     
%     for n = 1:this.N+1
%         Jx = Jx + (J{n})*y(offset(n):offset(n+1)-1);
%     end
%     for n = 1:this.N+1
%         y(offset(n):offset(n+1)-1) = (J{n})'*Jx;
%     end
    
    completeJac = [J{1},J{2},J{3},J{4}];
    gramian = completeJac'*completeJac;
    y = gramian*yold;
    
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
%     y = y1;
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
    JHJinv = pinv(this.gramian);
    y = JHJinv*q;    
end

function res = residFun(this,z)
    X = this.x;
    Y = this.y;
    U = z{1};
    S = z{2};
    
    Ux = cellfun(@(u) u'*X',U,'UniformOutput',false);
    Uxf = flip(Ux);
    Yest = S(:)'*kr(Uxf);
    
    res = Yest-Y;
end

function isvalid = validate(this, z)
    isvalid = true; % you can ignore this for now
end

function initialize(this, z) % initialize some things, e.g., cached variables.
    
end 

end

end