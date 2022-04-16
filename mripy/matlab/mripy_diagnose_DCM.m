function D = mripy_diagnose_DCM(GCM)
% Diagnose the goodness-of-fit of DCMs.
% 
% Return a table, in which each row is a DCM (for one subject) and the
% columns are:
% - explained: percent variance explained
% - LAPE: largest absolute posterior expectation (extrinsic connections)
% - complexity: complexity and effective number of parameters estimated
% - GOF: PC1 of all above as a more general measure of goodness-of-fit
% 
% See also: spm_dcm_fmri_check
% 2021-02-20: Created by qcc

    DCM = spm_dcm_fmri_check(GCM, true);
    D = cellfun(@(x)x.diagnostics, DCM, 'UniformOutput', false);
    D = cell2table(num2cell(cell2mat(D)), 'VariableNames', {'explained', 'LAPE', 'complexity'});
    X = zscore(table2array(D));
    X = X(:, ~(any(isnan(X)) | any(isinf(X)) | all(X==0)));
    [~, score] = pca(X, 'VariableWeights', 'variance');
    D.GOF = score(:,1); % PC1 as a more general measure of goodness-of-fit
end

