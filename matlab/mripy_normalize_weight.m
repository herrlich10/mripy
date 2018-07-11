function Q = mripy_normalize_weight(Q, exclude_diag)
%MRIPY_NORMALIZE_WEIGHT Normalize positive and negative weights to sum up to ±1
%   2017-08-06: Created by qcc
    if nargin < 2
        exclude_diag = false;
    end
    if exclude_diag
        Q(eye(length(Q))>0.5) = 0;
    end
    P = (Q > 0);
    N = (Q < 0);
    Q(P) = Q(P) / sum(Q(P(:)));
    Q(N) = Q(N) / abs(sum(Q(N(:))));
    % Sanity checks
    assert(abs(sum(Q(:))) < 1e-12);
    assert(abs(sum(Q(Q(:)>0))-1) < 1e-12);
end

