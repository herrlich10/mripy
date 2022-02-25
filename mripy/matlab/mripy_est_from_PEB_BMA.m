function est = mripy_est_from_PEB_BMA(BMA, norm)
% Extract estimates from a PEB BMA object.
%   est.A, est.B, est.C, est.PA, est.PB, est.PC
    if nargin < 2
        norm = false;
    end
    n_params = size(BMA.Pnames, 1);
    for k = 1:n_params
        if norm
            eval(['est.', BMA.Pnames{k}, ' = ', num2str(full(BMA.Ep(k,1)./sqrt(BMA.Cp(k,k)))), ';']);
        else
            eval(['est.', BMA.Pnames{k}, ' = ', num2str(full(BMA.Ep(k,1))), ';']);
        end
    end
    if isfield(est, 'C') && size(est.C, 1) == 1 && size(est.C, 2) ~= 1 %#ok<NODEF>
        est.C = est.C';
    end
    % Use Pp
    for k = 1:n_params
        eval(['est.P', BMA.Pnames{k}, ' = ', num2str(BMA.Pp(k,1)), ';']);
    end
    if isfield(est, 'PC') && size(est.PC, 1) == 1 && size(est.PC, 2) ~= 1 
        est.PC = est.PC';
    end
end

