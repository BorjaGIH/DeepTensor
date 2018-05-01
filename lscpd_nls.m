function [U,output] = lscpd_nls(A, b, U0, varargin)
%LSCPD_NLS LS-CPD by nonlinear least squares
%   [U,output] = lscpd_nls(A, b, U0, varargin) computes the solution of a
%   linear system with a (vectorized) CPD structure on the solution. U0
%   contains a CPD with U0{n} of size I_n x R. The matrix A can be given as
%   a matrix of size I_0 x I_1I_2...I_N or as a tensor of size I_0 x ...
%   I_1 x i_2 x ... x I_N. The algorithm uses U0 as initialization. The
%   structure output returns additional information:
%
%      output.Name  - The name of the selected algorithm.
%      output.<...> - The output of the selected algorithm.
%
%   lscpd_nls(A, b, x0, options) may be used to set the following options:
%
%      options.Algorithm =   - The desired optimization method.
%      [@nls_gncgs| ...
%       {@nls_gndl}|@nls_lm]
%      options.M =           - The preconditioner to use when
%      [{'block-Jacobi'}|...   options.LargeScale is true.
%       false]
%      options.<...>         - Parameters passed to the selected method,
%                              e.g., options.TolFun, options.TolX, ...
%                              See also help [options.Algorithm].

%   Authors: Martijn Boussé         (Martijn.Bousse@esat.kuleuven.be)
%            Nico Vervliet          (Nico.Vervliet@esat.kuleuven.be)
%            Lieven De Lathauwer    (Lieven.DeLathauwer@kuleuven-kulak.be)

    % check initial factor matrices U0
    U = U0(:).';
    N = length(U);
    R = size(U{1},2); 
    if any(cellfun('size',U,2) ~= R)
        error('lscpd_nls:U0','size(U0{n},2) should be the same for all n.');
    end
    
    % convert A to a tensor
    if ndims(A) ~= N + 1
        if prod(cellfun(@length, U)) ~= size(A, 2), 
            error('lscpd_nls:dimensionMismatch', 'Dimensions don''t agree');
        end
        A = reshape(A, [size(A,1); cellfun(@length, U(:))]');
    end
    nbrows = size(A,1);

    % permute the first mode to the middle to minimize middle contractions
    m1 = ceil(0.5*(N+1));
    A = permute(A, [2:m1 1 m1+1:N+1]);
    modes = [1:m1-1 m1+1:N+1];
    
    % create helper functions and variables
    isfunc = @(f)isa(f,'function_handle');
    xsfunc = @(f)isfunc(f)&&exist(func2str(f),'file');
    funcs = {@nls_gndl2,@nls_gncgs,@nls_lm}; % MODIFICATION BY BORJA VELASCO (nls_gndl2)
    
    % parse options
    p = inputParser;
    p.KeepUnmatched = true;
    p.addParameter('Algorithm', funcs{find(cellfun(xsfunc,funcs),1)});
    p.addParameter('CGMaxIter', 10);
    p.addParameter('Display', 0);
    p.addParameter('M', 'block-Jacobi');
    p.parse(varargin{:});
    options = [fieldnames(p.Results)'  fieldnames(p.Unmatched)';
               struct2cell(p.Results)' struct2cell(p.Unmatched)'];
    options = struct(options{:});
        
    % cache variables
    cache.offset = cellfun(@(u) size(u,1),U(:));
    cache.offset = cumsum([1; kron(cache.offset,ones(R,1))]);
    
    % call the optimization method.
    dF.JHJx = @JHJx;
    dF.JHF = @grad;
    switch options.M
      case 'block-Jacobi', dF.M = @M_blockJacobi;
      otherwise, if isfunc(options.M), dF.M = options.M; end
    end
    [U,output] = options.Algorithm(@objfun,dF,U(:).',options);
    output.Name = func2str(options.Algorithm);

    function fval = objfun(z)
    % LS-CPD objective function
        fval = -b;
        for r = 1:R
            tmp = cellfun(@(u) u(:,r), z, 'UniformOutput', false);
            fval = fval + contract(A, tmp, modes);
        end
        cache.residual = fval;
        fval = 0.5*(fval(:)'*fval(:));
    end

    function grad = grad(z)
    % LS-CPD scaled conjugate cogradient.
        E = cache.residual;
        offset = cache.offset;
        grad = nan(offset(end)-1,1);
        
        for n = 1:N
            for r = 1:R
                tmp = cellfun(@(u) u(:,r), z, 'UniformOutput', false);
                cache.J{r}{n} = contract(A, tmp([1:n-1 n+1:N]), modes([1:n-1 n+1:N]));                
                if n > N/2
                    cache.J{r}{n} = cache.J{r}{n}.';
                end                   
                grad(offset(r+(n-1)*R):offset(r+1+(n-1)*R)-1) = conj(cache.J{r}{n})*E;
                cache.JHJinv{r}{n} = pinv(conj(cache.J{r}{n})*cache.J{r}{n}.');
            end
        end             
    end

    function y = JHJx(~,y)
    % LS-CPD fast Gramian vector product
        offset = cache.offset;
        J  = cache.J;
        Jx = zeros(nbrows, 1);
        for n = 1:N
            for r = 1:R
                Jx = Jx + (y(offset(r+(n-1)*R):offset(r+1+(n-1)*R)-1).'*J{r}{n}).';
            end
        end
        for n = 1:N
            for r = 1:R
                y(offset(r+(n-1)*R):offset(r+1+(n-1)*R)-1) = conj(J{r}{n})*Jx;
            end
        end
    end

    function y = M_blockJacobi(~,q)
    % Solve M*y = q, where M is a block-diagonal approximation for JHJ.
        y = nan(size(q));
        offset = cache.offset;
        for n = 1:N
            for r = 1:R
                idx = offset(r+(n-1)*R):offset(r+1+(n-1)*R)-1;
                y(idx) = cache.JHJinv{r}{n}*q(idx);
            end
        end
    end
   
end