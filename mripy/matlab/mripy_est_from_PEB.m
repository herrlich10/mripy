function est = mripy_est_from_PEB(PEB, norm)
% Extract estimates from a PEB object.
%   est.A, est.B, est.C, est.PA, est.PB, est.PC
    if nargin < 2
        norm = false;
    end
    n_params = size(PEB.Pnames, 1);
    for k = 1:n_params
        if norm
            eval(['est.', PEB.Pnames{k}, ' = ', num2str(full(PEB.Ep(k,1)./sqrt(PEB.Cp(k,k)))), ';']);
        else
            eval(['est.', PEB.Pnames{k}, ' = ', num2str(full(PEB.Ep(k,1))), ';']);
        end
    end
    if isfield(est, 'C') && size(est.C, 1) == 1 && size(est.C, 2) ~= 1 %#ok<NODEF>
        est.C = est.C';
    end
    % Compute uncorrected p-value from z-score
    for k = 1:n_params
        z = full(PEB.Ep(k,1)./sqrt(PEB.Cp(k,k)));
        if z > 0
            p = 2*(1-normcdf(z));
        else
            p = 2*normcdf(z);
        end
        eval(['est.P', PEB.Pnames{k}, ' = ', num2str(1-p), ';']);
    end
    if isfield(est, 'PC') && size(est.PC, 1) == 1 && size(est.PC, 2) ~= 1 
        est.PC = est.PC';
    end
end

