function est = mripy_est_from_GCM(GCM, norm)
% Extract estimates from a GCM object.
%   est.A, est.B, est.C, est.PA, est.PB, est.PC
    if nargin < 2
        norm = false;
    end
    if norm % Ep./sqrt(Cp)
        DCM = GCM{1,1};
        index = [];
        count = 0;
        for x = 'abc'
            idx = (1:numel(DCM.(x)))';
            index.(upper(x)) = count + idx(DCM.(x)>0.5);
            count = count + numel(DCM.(x));
        end
        est.GA = cell2mat(cellfun(@(x)shiftdim(Ep_div_Cp(x,'A',index), -1), GCM, 'UniformOutput', false));
        est.GB = cell2mat(cellfun(@(x)shiftdim(Ep_div_Cp(x,'B',index), -1), GCM, 'UniformOutput', false));
        est.GC = cell2mat(cellfun(@(x)shiftdim(Ep_div_Cp(x,'C',index), -1), GCM, 'UniformOutput', false));
    else
        est.GA = cell2mat(cellfun(@(x)shiftdim(x.Ep.A, -1), GCM, 'UniformOutput', false));
        est.GB = cell2mat(cellfun(@(x)shiftdim(x.Ep.B, -1), GCM, 'UniformOutput', false));
        est.GC = cell2mat(cellfun(@(x)shiftdim(x.Ep.C, -1), GCM, 'UniformOutput', false));
    end
    est.A = squeeze(mean(est.GA, 1));
    est.B = squeeze(mean(est.GB, 1));
    est.C = squeeze(mean(est.GC, 1));
    if size(est.C, 1) == 1 && size(est.C, 2) ~= 1
        est.C = est.C';
    end
    [~,p] = ttest(est.GA);
    est.PA = squeeze(1 - p);
    [~,p] = ttest(est.GB);
    est.PB = squeeze(1 - p);
    [~,p] = ttest(est.GC);
    est.PC = squeeze(1 - p);
    if size(est.PC, 1) == 1 && size(est.PC, 2) ~= 1
        est.PC = est.PC';
    end
end


function Ep = Ep_div_Cp(DCM, X, index)
    Ep = DCM.Ep.(X);
    Cp = full(DCM.Cp(sub2ind(size(DCM.Cp), index.(X), index.(X))));
    I = DCM.(lower(X))>0.5;
    Ep(I) = Ep(I)./sqrt(Cp); % Fixed bug (20220415): miss sqrt
end