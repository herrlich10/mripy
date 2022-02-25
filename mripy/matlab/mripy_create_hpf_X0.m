function X0 = mripy_create_hpf_X0(n, RT, constant)
% Create sine wave regressors for high-pass filtering.
%
% See also: spm_filter, spm_dctmtx
% 2021-02-19: Created by qcc

    global defaults
    if nargin < 3
        constant = 'none';
    end
    HParam = defaults.stats.fmri.hpf; % 128 s
    k = fix(2*(n*RT)/HParam + 1);
    X0 = spm_dctmtx(n, k);
    switch constant
        case 'none'
            X0 = X0(:,2:end);
        case 'one'
            X0 = [ones(size(X0,1),1), X0(:,2:end)];
        case 'dct'
        case 'spherical'
            K = X0(:,2:end);
            W = 1; % We cannot estimate non-sphericity without seeing raw ROI data
            X = W*ones(size(X0,1),1);
            X = X - K*(K'*X); % Project one to the complement of K (i.e., regress out K from vector 1)
            X0 = [X, K];
        case 'non-spherical'
            N = size(X0,1);
            K = X0(:,2:end);
            W = eye(N) + diag(ones(N-1,1)*-0.1, 1) + diag(ones(N-1,1)*-0.1, -1); % Fake W
            X = W*ones(N,1);
            X = X - K*(K'*X); % Project one to the complement of K (i.e., regress out K from vector 1)
            X0 = [X, K];
        case 'spm'
            error('NotImplementedError');
    end
end

