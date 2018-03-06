classdef TensorOptimizationKernel < handle & matlab.mixin.CustomDisplay
%TENSOROPTIMIZATIONKERNEL Generic optimization kernel for tensor problems.
%   TENSOROPTIMIZATIONKERNEL is an abstract class defining the template for
%   tensor optimization problems. This template consists of generic properties
%   and functions.
%   
%   TENSOROPTIMIZATIONKERNEL is an internal class which is indented as a contract
%   for specific kernels to allow re-use across different parts of Tensorlab. To
%   access more advanced information, type
%   
%       edit TensorOptimizationKernel
%   
%   or
%   
%       doc TensorOptimizationKernel
%   
%   See also: CPDKernel, CPDIKernel, CPDLIKernel
     
%   TENSOROPTIMIZATIONKERNEL defines the following functions that have to be
%   implemented:
%
%       - A constructor             create kernel 
%       - initialize(z)             initialize kernel 
%       - objfun(z)                 compute objective function value
%       - grad(z)                   compute gradient
%       - validate(z)               validate parameters, data and variables
%
%   The following functions are optional for quasi-Newton methods, but
%   required when using NLS methods (Gauss-Newton and Levenberg-Marquardt):
%
%       - JHDJ(z)                   Gramian (approximation of the Hessian)
%       - JHDJx(z,x)                Gramian (approximation of the Hessian)
%
%   The following functions are always optional
%
%       - state(z)                  Update the iteration dependent state 
%       - preaction(z)              Action performed in the beginning of an
%                                   iteration 
%       - postaction(z)             Action performed at the end of an
%                                   iteration. 
%       
%   Kernels can additionally implement other functions such as
%   preconditioners, e.g. 
%
%       function x = M_Jacobi(this, z, b)
%   
%   To implement a specific kernel, define a new class that inherits from
%   TENSOROPTIMIZATIONKERNEL
%
%       classdef SpecificKernel < TensorOptimizationKernel
%       
%       properties
%           % Add data, cached variables, set properties, ...
%       end 
%
%       methods
%
%           function this = SpecificKernel(data, varargin)
%               % Constructor: this is the place to set parameters, data,
%               % etc. 
%           end 
%
%           % Add implementations of necessary functions 
%       end    
%
%   Note that all functions have the current kernel as first parameter (which
%   we usually call 'this'), e.g.
%
%       function fval = objfun(this, z)
%
%   To use a specific kernel, e.g. CPDKernel directly, the kernel is
%   initialized, options are set, initialize is called and an optimization
%   algorithm is run. For example:
%
%       % Construct kernel with options 
%       kernel = CPDKernel(T);
%       % Initialize kernel
%       kernel.initialize(U0);
%       % Call optimization routine
%       fval    = @(z) kernel.objfun(z);
%       dF.JHF  = @(z) kernel.grad(z);
%       dF.JHJx = @(z,x) kernel.JHDJx(z,x);
%       dF.M    = @(z,b) kernel.M_blockJacobi(z,b);
%       [Ures, output] = nls_gndl(fval, dF, U0);
    
% Author(s):  Nico Vervliet       (Nico.Vervliet@kuleuven.be)
%             Lieven De Lathauwer (Lieven.DeLathauwer@kuleuven.be)
%
% Version History:
% - 2018/02/07   NV      Initial version.
    
    properties

        %% Abilities of a the kernel
        % SDF can dispatch some tasks such as dealing with constants and
        % symmetry to the kernel, as they can be handled more efficiently
        % there. 
        
        % The kernel can handle symmetry
        handlesSymmetry = false     
        % The kernel can handle constants
        handlesConstants = false
    end 
    
    properties 
        
        %% Abilities of a the kernel

        % The data is symmetric (only relevant if handlesSymmetry = true)
        isDataSymmetric = false
        % Array of length at least the order of the data indicating which
        % variable is used in each mode (only relevant if handlesSymmetry =
        % true)
        variablesInMode

        % Logicals indicating that a variables is constant ((cell of) logical
        % vectors, e.g., [1 0 0] for a CPD or {[0 0 1 0], [0 0 0 0]} for a
        % BTD.) (Only relevant if handlesConstants is true)
        isConstant
        
        %% Persistent storage
        % If enabled, keep a copy of the variables every iteration. This
        % action should be implemented in POSTACTION. 
        
        % Enable persistent routines. 
        makePersistent = false
        % Persistent copy of the variables (only valid if makePersistent =
        % true and this variable is set during POSTACTION.) 
        Z
    end 
    
    methods (Abstract)
    
        initialize(this, z);
        %INITIALIZE Initialize this kernel.
        %   INITIALIZE(Z) initializes the kernel by precomputing information independent
        %   of the iterations, and should be called before the optimization
        %   algorithm is run. INITIALIZE can only depend on the properties of the
        %   variables and not the actual values, e.g., information on sizes and
        %   whether or not the variables are complex can be used.
                
        fval = objfun(this, z);
        %OBJFUN Compute objective function value. 
        %   FVAL = OBJFUN(Z) computes the objective function value in the variables Z.
        
        grad = grad(this, z);
        %GRAD Compute gradient.
        %   GRAD = GRAD(Z) computes the gradient GRAD of the objective function in the
        %   variables Z. 

        isvalid = validate(this, z);
        %VALIDATE Validate the kernel.
        %   ISVALID = VALIDATE(Z) checks if the kernel is valid, i.e., if this
        %   particular combination of paramaters, data and variables Z allows
        %   all kernel routines to be completed without errors (unless
        %   underlying routines defined elsewhere fail.)
     
    end 
    
    methods
        
        function state(this, z)
        %STATE Update iteration-dependent state.              
        %   STATE(Z) precomputes or updates information that depends on the variables in
        %   an iteration, and is primarely used to update cached
        %   variables. STATE should be called by the user any time the variables
        %   change. A good place to do this, is at the beginning of the gradient
        %   or at the function evaluation.
            
        % Default implementation does nothing. 
        end 
        
        function jhj = JHDJ(this, z)
        %JHDJ Compute an approximation of the Hessian. 
        %   JHJ = JHDJ(Z) computes a positive semidefinite approximation of the Hessian
        %   of the objective function using variables Z. This approximation is
        %   usually the Gramian matrix, or its generalization to other
        %   divergences.
             
            error('tensorlab:notimplemented', ...
                  'JHDJ is not implemented in this kernel');
        end 

        function y = JHDJx(this, z, x)
        %JHDJX Compute the approximate Hessian vector product. 
        %   Y = JHDJX(Z,X) computes the product of the positive semidefinite
        %   approximation of the Hessian of the objective function constructed
        %   using variables Z, with a vector X. The approximation is usually the
        %   Gramian matrix, or its generalization to other divergences.

            error('tensorlab:notimplemented', ...
                  'JHDJx is not implemented in this kernel');
        end

        function z = preaction(this, z)
        %PREACTION Perform actions on z at the end of every iteration. 
        %   Z = PREACTION(Z) performs actions on Z before every iteration, or can be
        %   used to display information etc.

        % Default implementation does nothing.
        end 
        
        function z = postaction(this, z)
        %POSTACTION Perform actions on z at the end of every iteration. 
        %   Z = POSTACTION(Z) performs actions on Z after every iteration, or can be
        %   used to display information etc.
            
        % Update current variables if persistent is true 
            if this.makePersistent 
                this.Z = z;
            end 
            
        end 
    end 
    
    % Auxiliary functions for writing tests. 
    %    The tests are executed using the XUnit framework, which should be
    %    loaded first. 
    %
    %    Note: these functions can be expensive, so use small examples. 
    methods 

        function test_grad(this, z, tol)
        %TEST_GRAD Test correctness GRAD using finite differencs.
        %   TEST_GRAD(Z) tests if the gradient GRAD(Z) of the objective function
        %   of this kernel is correct when comparing with the numerically
        %   computed gradient (using finite differences.)
        %   
        %   See also: test_JHDJ, test_JHDJx
            
            if nargin < 3, tol = 1e-6; end
            
            this.validate(z);      % Should not fail here
            this.initialize(z);    % Should not fail here
            this.objfun(z);        % Should not fail here (call required for state)
            
            % target
            grad1 = deriv(@this.objfun, z, this.objfun(z), 'gradient');
            grad1 = TensorOptimizationKernel.serialize(grad1);
            % computed
            grad2 = this.grad(z); 
            
            % Check if correct
            relerr = frob(grad1-grad2)/frob(grad1);
            if exist('assertElementsAlmostEqual.m', 'file')
                assertElementsAlmostEqual(relerr, 0, 'absolute', tol);
            else % fall back to Matlab's Unit Test Framework
                assert(relerr <= tol);
            end 
        end 
        
        function test_JHDJ(this, z, model, fun, elementwise, tol)
        %TEST_JHDJ Test correctness JHJD using finite differences.
        %   TEST_JHDJ(Z, MODEL) tests if the Gramian of a LS approximation of a
        %   tensor T by MODEL(Z) is correct. The Gramian is computed using
        %   numerical derivatives. The variables are Z and MODEL is a function
        %   handle that creates a tensor from Z, e.g., in the case of a CPD,
        %   MODEL = @(Z) cpdgen(Z).
        %
        %   TEST_JHDJ(Z, MODEL, FUN) tests if the JHDJ approximation of the
        %   Hessian is correct. The behavior depends on FUN:
        %   - If FUN is numeric, it should contain the second-order derivative
        %     of the objective function w.r.t. the model. For example, in the
        %     case of a LS loss function 0.5*||M-T||^2, with model M and tensor
        %     T, FUN = eye(numel(T)). In the case the derivative is diagonal,
        %     providing the diagonal suffices. If this diagonal is constant,
        %     providing this constant value suffices. Hence, in the case of a LS
        %     loss functions, FUN can be ones(numel(MZ),1) or simply 1.
        %   - Otherwise, FUN should be a function, i.e., FUN = @(M,i) OBJ(m,i),
        %     in which OBJ is the objective function value contribution of the
        %     ith tensor entry. This assumes that the total objective
        %     function can be written as 
        %
        %         fval = sum_i f_i(m(i))
        % 
        %     in which f_i is OBJ(m,i) and m(i) the ith entry of the tensor
        %     generated by the model. For example, in the case of the LS loss
        %     function, f_i = OBJ(m,i) = 0.5 * (m(i) - t(i))^2. 
        %
        %   TEST_JHDJ(Z, MODEL, FUN, FALSE) assumes FUN is not numeric and
        %   cannot be written as a sum of contributions for each tensor
        %   entry. FUN should be a function computing the objective function
        %   value given M = RESHAPE(MODEL(Z),[],1), i.e., the vectorized,
        %   evaluated model. For example, for the LS loss function,
        % 
        %       FUN = @(M) 0.5*frob(M-T(:))^2
        %
        %   The fourth argument is actual ELEMENTWISE, which defaults to true.
        %
        %   TEST_JHDJ(Z, MODEL, FUN, ELEMENTWISE, TOL) sets an absolute
        %   tolerance on the relative error computed using the Frobenius
        %   norm. Default: 1e-5. 
        %
        %   Note thate computing second-order derivatives (FUN) numerically is
        %   inaccurate. If possible, provide FUN by evaluating the analytically
        %   derived second-order derivative.
        %
        %   Note that computing second-order derivatives is computationally
        %   expensive if ELEMENTWISE = FALSE. 
        %
        %   See also: test_JHDJx, test_grad
            
            if nargin < 4, fun = 1; end
            if nargin < 5, elementwise = isnumeric(fun) || nargin(fun) == 2; end
            if nargin < 6, tol = 1e-5; end 
            
            this.validate(z);       % Should not fail here
            this.initialize(z);     % Should not fail here
            this.objfun(z);         % Should not fail here (call required for state)
            this.grad(z);           % Should not fail here (call required for state)
            
            % target
            J1 = deriv(model, z, [], 'Jacobian');
            M  = reshape(model(z), [], 1);
            if isnumeric(fun) 
                D1 = fun;
                if isvector(D1), D1 = diag(D1); end
            else
                D1 = TensorOptimizationKernel.numericHessianFD(fun, M, elementwise);
            end 
            JHDJ1 = J1'*D1*J1;
            
            % computed
            JHDJ2 = this.JHDJ(z);
            
            % Check if correct
            relerr = frob(JHDJ1-JHDJ2)/frob(JHDJ1);
            if exist('assertElementsAlmostEqual.m', 'file')
                assertElementsAlmostEqual(relerr, 0, 'absolute', tol);
            else % fall back to Matlab's Unit Test Framework
                assert(relerr <= tol);
            end 
        end 
        
        function test_JHDJx(this, z, x, model, fun, elementwise, tol)
        %TEST_JHDJX Test correctness JHJDX using finite differences.
        %   TEST_JHDJX(Z, X, MODEL) tests if the Gramian-vector product of a LS
        %   approximation of a tensor T by MODEL(Z) is correct. The Gramian is
        %   computed using numerical derivatives. The variables are Z and MODEL
        %   is a function handle that creates a tensor from Z, e.g., in the case
        %   of a CPD, MODEL = @(Z) cpdgen(Z).
        %
        %   TEST_JHDJX(Z, X, MODEL, FUN) tests if the JHDJX approximation of the
        %   Hessian-vector product is correct. The behavior depends on FUN:
        %   - If FUN is numeric, it should contain the second-order derivative
        %     of the objective function w.r.t. the model. For example, in the
        %     case of a LS loss function 0.5*||M-T||^2, with model M and tensor
        %     T, FUN = eye(numel(T)). In the case the derivative is diagonal,
        %     providing the diagonal suffices. If this diagonal is constant,
        %     providing this constant value suffices. Hence, in the case of a LS
        %     loss functions, FUN can be ones(numel(MZ),1) or simply 1.
        %   - Otherwise, FUN should be a function, i.e., FUN = @(M,i) OBJ(m,i),
        %     in which OBJ is the objective function value contribution of the
        %     ith tensor entry. This assumes that the total objective
        %     function can be written as 
        %
        %         fval = sum_i f_i(m(i))
        % 
        %     in which f_i is OBJ(m,i) and m(i) the ith entry of the tensor
        %     generated by the model. For example, in the case of the LS loss
        %     function, f_i = OBJ(m,i) = 0.5 * (m(i) - t(i))^2. 
        %
        %   TEST_JHDJX(Z, X, MODEL, FUN, FALSE) assumes FUN is not numeric and
        %   cannot be written as a sum of contributions for each tensor
        %   entry. FUN should be a function computing the objective function
        %   value given M = RESHAPE(MODEL(Z),[],1), i.e., the vectorized,
        %   evaluated model. For example, for the LS loss function,
        % 
        %       FUN = @(M) 0.5*frob(M-T(:))^2
        %
        %   The fourth argument is actual ELEMENTWISE, which defaults to true.
        %
        %   TEST_JHDJX(Z, X, MODEL, FUN, ELEMENTWISE, TOL) sets an absolute
        %   tolerance on the relative error computed using the Frobenius
        %   norm. Default: 1e-5.
        %
        %   Note thate computing second-order derivatives (FUN) numerically is
        %   inaccurate. If possible, provide FUN by evaluating the analytically
        %   derived second-order derivative.
        %
        %   Note that computing second-order derivatives is computationally
        %   expensive if ELEMENTWISE = FALSE. 
        % 
        %   See also: test_JHDJ, test_grad
            
            if nargin < 5, fun = 1; end
            if nargin < 6, elementwise = isnumeric(fun) || nargin(fun) == 2; end
            if nargin < 7, tol = 1e-5; end 
            
            this.validate(z);       % Should not fail here
            this.initialize(z);     % Should not fail here
            this.objfun(z);         % Should not fail here (call required for state)
            this.grad(z);           % Should not fail here (call required for state)
            
            % target
            J1 = deriv(model, z, [], 'Jacobian');
            M  = reshape(model(z), [], 1);
            if isnumeric(fun) 
                D1 = fun;
                if isvector(D1), D1 = diag(D1); end
            else
                D1 = TensorOptimizationKernel.numericHessianFD(fun, M, elementwise);
            end 
            y1 = J1'*(D1*(J1*x));
            
            % computed
            y2 = this.JHDJx(z,x);
            
            % Check if correct
            relerr = frob(y1-y2)/frob(y1);
            if exist('assertElementsAlmostEqual.m', 'file')
                assertElementsAlmostEqual(relerr, 0, 'absolute', tol);
            else % fall back to Matlab's Unit Test Framework
                assert(relerr <= tol);
            end 
        end 
    end 

    methods (Static)
        
        function z = serialize(z)
            if iscell(z)
                for i = find(cellfun(@iscell,z(:).'))
                    z{i} = TensorOptimizationKernel.serialize(z{i});
                end
                s = cellfun(@numel,z(:)); o = [0; cumsum(s)];
                c = z; z = zeros(o(end),1);
                for i = 1:length(s), z(o(i)+(1:s(i))) = c{i}(:); end
            else
                z = z(:);
            end
        end
        
        function D = numericHessianFD(f, m, elementwise)
        %NUMERICHESSIANFD Numerically compute Hessian using finite differences.
        %   D = NUMERICHESSIANFD(F,M) computes the Hessian of a function F in M using
        %   finite differences, in which F is the objective function as a
        %   function handle with the model M as parameter, i.e., D(i,j) = d^2F/
        %   (dM(i)*dM(j)). In the case the objective function can be written as
        %   a sum of contributions of each tensor entry, i.e.,
        %
        %       fval = sum_i f_i(m(i))
        % 
        %   in which f_i is ith objective OBJ(m,i) and m(i) the ith entry of the
        %   tensor generated by the model, F = @(m,i) OBJ(m,i) and D is
        %   diagonal. For example, in the case of the LS loss function, 
        %
        %       f_i = 0.5 * (m(i) - t(i))^2.
        % 
        %   D = NUMERICHESSIANFD(F,M,ELEMENTWISE) computes the above if
        %   ELEMENTWISE = TRUE, otherwise, if ELEMENTWISE = FALSE, D is not
        %   diagonal. In this case, F is a function of the vectorized model M,
        %   i.e., F = @(M) OBJ(M). For example, in the case of the LS loss
        %   function,
        %
        %       F = @(M) 0.5*frob(M - T(:))^2
            
            if nargin < 3, elementwise = nargin(f) == 2; end
                        
            if elementwise
                D = zeros(numel(m),1);
                e = sqrt(sqrt(eps(class(m))));
                for k = 1:numel(m)
                    p = e*max(1,abs(real(m(k))));
                    D(k) = (f(m(k)+p,k)-2*f(m(k),k)+f(m(k)-p,k))/p^2;
                end 
                D = diag(D);
            else 
                D = zeros(numel(m));
                e = sqrt(sqrt(eps(class(m))));
                p1 = zeros(numel(m),1);
                p2 = zeros(numel(m),1);
                for k = 1:numel(m) 
                    if k > 1, p1(k-1) = 0; end 
                    p1(k) = e*max(1,abs(real(m(k))));
                    for l = k:numel(m)
                        if l > 1, p2(l-1) = 0; end
                        p2(l) = e*max(1,abs(real(m(l))));
                        
                        d11 = f(m+p1+p2);
                        d12 = f(m+p1-p2);
                        d21 = f(m-p1+p2);
                        d22 = f(m-p1-p2);
                        
                        D(k,l) = (d11 - d12 - d21 + d22)/(4*p1(k)*p2(l));
                        D(l,k) = D(k,l);
                    end 
                end 
            end 
        end 
        
    end 
    
end 
