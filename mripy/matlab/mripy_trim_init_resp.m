function trimmed = mripy_trim_init_resp(x, trim, run_len)
% Trim the initial TRs of each run.
% 2021-03-05: Created by qcc
    assert(mod(length(x), run_len)==0);
    n_runs = length(x) / run_len;
    trimmed = reshape(x, run_len, n_runs);
    trimmed = trimmed(1+trim:end,:);
    trimmed = trimmed(:);
    if size(x, 1) == 1
        trimmed = trimmed';
    end
end

