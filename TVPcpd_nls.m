function [U,output] = TVPcpd_nls(X, b, U0, varargin)
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

%   Modification by Borja Velasco 12/5/18


    % check initial factor matrices U0
    U = U0(:).';
    N = length(U);
    R = size(U{1},2); 
    dim = size(U{1},1);
    
    if any(cellfun('size',U,2) ~= R)
        error('lscpd_nls:U0','size(U0{n},2) should be the same for all n.');
    end
    
    % convert A to a tensor
%     if ndims(A) ~= N + 1
%         if prod(cellfun(@length, U)) ~= size(A, 2), 
%             error('lscpd_nls:dimensionMismatch', 'Dimensions don''t agree');
%         end
%         A = reshape(A, [size(A,1); cellfun(@length, U(:))]');
%     end
%     nbrows = size(A,1);

    % permute the first mode to the middle to minimize middle contractions
%     m1 = ceil(0.5*(N+1));
%     A = permute(A, [2:m1 1 m1+1:N+1]);
%     modes = [1:m1-1 m1+1:N+1];
    
    % create helper functions and variables
    isfunc = @(f)isa(f,'function_handle');
    xsfunc = @(f)isfunc(f)&&exist(func2str(f),'file');
    funcs = {@nls_gndl,@nls_gncgs,@nls_lm};
    
    % parse options
    p = inputParser;
    p.KeepUnmatched = true;
    p.addParameter('Algorithm', funcs{find(cellfun(xsfunc,funcs),1)});
    p.addParameter('CGMaxIter', 10);
    p.addParameter('Display', 0);
    p.addParameter('M', 'block-Jacobi');
%     p.addParameter('M', 'M'); % Modified until fast multiplication is implemented
    p.parse(varargin{:});
    options = [fieldnames(p.Results)'  fieldnames(p.Unmatched)';
               struct2cell(p.Results)' struct2cell(p.Unmatched)'];
    options = struct(options{:});
        
    % cache variables
    cache.offset = cellfun(@(u) numel(u),U(:));
    cache.offset = cumsum([1; kron(cache.offset,1)]);
    
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
        % Tensor-vector product LS-CPD objective function
        tmp = cellfun(@(u) X*u, z, 'UniformOutput', false);
        mult = tmp{1};
        for k = 2:length(tmp)
            mult = mult.*tmp{k};
        end
        cache.residual = mult*ones(R,1)-b;
        fval = cache.residual;
        fval = 0.5*(fval(:)'*fval(:));
    end

    function grad = grad(z)
        % Tensor-vector product LS-CPD gradient
        E = cache.residual;
        grad = [];
        
        for n = 1:N
            for ii = 1:R*dim
                zaux = z;
                zaux{n} = zeros(size(zaux{n}));
                zaux{n}(ii) = 1;
                tmp = cellfun(@(u) X*u, zaux, 'UniformOutput', false);
                mult = tmp{1};
                for k = 2:N
                    mult = mult.*tmp{k};
                end
                cache.J{n}{ii} = mult*ones(R,1);
                grad = [grad, cache.J{n}{ii}'*E];
%                 cache.JHJinv{n}{ii} = pinv(cache.J{n}{ii}'*cache.J{n}{ii});
            end
        end
        grad = grad';
%         grad2 = grad;
        J = cell2mat(cellfun(@(u) cell2mat(u),cache.J,'UniformOutput', false));
        cache.JHJinv = pinv(J*J.');
    end

    function y = JHJx(~,y)
        % Tensor-vector product LS-CPD Gramian vector product. Implement fast via blocks
%         offset = cache.offset;
        J  = cache.J;
        
        JHJ = cell2mat(cellfun(@(u) cell2mat(u),J,'UniformOutput', false)).'*...
            cell2mat(cellfun(@(u) cell2mat(u),J,'UniformOutput', false));
        y = JHJ*y; 
    end

    function y = M_blockJacobi(~,q)
    % Solve M*y = q, where M is a block-diagonal approximation for JHJ.
%         y = nan(size(q));
%         y = [];
%         offset = cache.offset;
%         for n = 1:N
%             for ii = 1:R*dim
%                 idx = offset(n):offset(n+1)-1;
% %                 y(idx) = cache.JHJinv{n}{ii}*q(idx);
%                 y = [y, cache.JHJinv{n}{ii}*q(idx)];
%             end
%         end
        y = cache.JHJinv*q;
    end
   
end