function est = mripy_est_from_PEB_BMA(BMA, norm, effect)
% Extract estimates from a PEB BMA object.
%   est.A, est.B, est.C, est.PA, est.PB, est.PC
%   
%   norm : bool
%       Whether to show standardized z value as edge width in the graph.
%           true: show z value = mean/std
%           false: show original value = mean
%   effect : int
%       Column index in the group design matrix
%       specifying which group effect do we want to plot.
%           1: Mean
%           2: Covariate 1, ...
%   
% 2024-03-27: Add support to display additional effects (beyond group mean)
    if nargin < 2
        norm = false;
    end
    if nargin < 3
        effect = 1; % group mean: the 1 in BMA.Xnames
    end
    n_params = size(BMA.Pnames, 1);
    for k = 1:n_params
        K = k + n_params*(effect-1);
        if norm
            eval(['est.', BMA.Pnames{k}, ' = ', num2str(full(BMA.Ep(K,1)./sqrt(BMA.Cp(K,K)))), ';']);
        else
            eval(['est.', BMA.Pnames{k}, ' = ', num2str(full(BMA.Ep(K,1))), ';']);
        end
    end
    if isfield(est, 'C') && size(est.C, 1) == 1 && size(est.C, 2) ~= 1 %#ok<NODEF>
        est.C = est.C';
    end
    % Use Pp
    for k = 1:n_params
        K = k + n_params*(effect-1);
        eval(['est.P', BMA.Pnames{k}, ' = ', num2str(BMA.Pp(K,1)), ';']);
    end
    if isfield(est, 'PC') && size(est.PC, 1) == 1 && size(est.PC, 2) ~= 1 
        est.PC = est.PC';
    end
end

