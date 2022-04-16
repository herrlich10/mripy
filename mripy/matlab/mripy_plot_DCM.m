function [dcm, est, vis] = mripy_plot_DCM(dcm, est, vis)
% Plot the general DCM results (A, B, C, etc.)
% 
%   dcm: Basic information
%   ----------------------
%   The following are the only thing that you must specify for each plot:
%
%   dcm.a: The 0/1 structure of connectivity
%   dcm.b: The 0/1 existence of modulatory effect
%   dcm.c: The 0/1 existence of driving input
%   dcm.labels: The name of each node
%   dcm.layout: The xy coordinates of each node
%
%   (Optional)
%   dcm.node_size: The circle size of nodes. Default 10
%   dcm.angle_offset: Angular distance between forward/backward arrows. Default pi/12
%   dcm.font_size: Font size for node labels. Default 18
%   dcm.P_th: Significance threshold for posterior probability. Default 0.95
%
%   est: Esitmated parameters (optional)
%   ------------------------------------
%   If est is not provided, this function will generate a schematic diagram
%   illustrating the structure of the model. The linewidth/markersize of the
%   connections will not be scaled according to the estimated parameters.
%
%   est.A: Estimated parameter values
%   est.B:
%   est.C:
%   est.PA: Posterior probability of parameters (may also be 1-p for ttest)
%   est.PB:
%   est.PC:
%   
%   (Optional)
%   est.Amax, est.Amin, est.Bmax, est.Bmin, est.Cmax, est.Cmin: 
%       Value limits for normalizing visual element size (linewidth, markersize, etc.)
%       These can be manually set if you want, e.g., B and C share the same
%       scale for linewidth. 
%   est.normA, est.normB, est.normC: 
%       Normalization function x_norm = normX(x), default is @(x)(x-Xmin)./(Xmax-Xmin)
%
%   vis: Visual elements (color, style, alpha, location, show/hide, etc., optional)
%   -------------------------------------------------------------------------------
%   This optional argument allow you to customize the look and feel of the plot.
%
%   vis.mode: 'standard' (default) | 'single-modulatory'
%       Select which one of two kinds of plot will be generatd.
%       'standard' plots as much information as possible about A, B, C in one figure.
%           A and C are plotted as arrows, B are plotted as dots.
%           Multiple inputs are shown as different pairs of colors.
%           This is a quicker way to show all results at once.
%       'single-modulatory' only plots a single modulatory effect, i.e., B(:,:,vis.input)
%           One selected B is plotted as arrows.
%           This is more suitable for final publication as it is clean and clear.
%   vis.input: int
%       Select which input to plot in 'single-modulatory' mode.
%   vis.conn_colors:
%   vis.conn_styles:
%   vis.conn_alphas:
%   vis.input_colors:
%   vis.input_styles:
%   vis.max_width: Max linewidth for connections
%   vis.min_width: Min linewidth for connections
%   vis.mod_offset:
%   vis.show_ns_BC:
%   vis.show_legend: Whether to show legend for linewidth and color ('single-modulatory' mode only)
%   vis.conn_values: Text label to show for each connection ('single-modulatory' mode only)
%       The values can be different from est.B(:,:,vis.input), which are used for linewidth.
%       Usually, we use z value for linewidth, and actual value for labels.
%   vis.conn_label_x:
%   vis.conn_label_y:
%
%   EXAMPLES
%   --------
%   % Plot model structure
%   dcm.node_size = 7;
%   dcm.font_size = 10;
%   dcm.labels = {'V1', 'V4', 'aIPS', 'Pul'};
%   dcm.layout = [0,0; sqrt(3),1; 0,2; -2,1]*dcm.node_size*1.5;
%   dcm.a = GCM{1,1}.a;
%   dcm.b = GCM{1,1}.b;
%   dcm.c = GCM{1,1}.c;
%   mripy_plot_DCM(dcm);
%
%   % Plot simple group mean for A, B, C
%   est1 = mripy_est_from_GCM(GCM);
%   vis.show_ns_BC = true;
%   mripy_plot_DCM(dcm, est1, vis);
%
%   % Plot PEB BMA results for B(:,:,2)
%   est2 = mripy_est_from_PEB_BMA(BMA_B_search);
%   est3 = mripy_est_from_PEB_BMA(BMA_B_search, true);
%   vis = struct('mode', 'single-modulatory', 'input', 2);
%   vis.conn_values = est2.B(:,:,vis.input);
%   [~,~,vis] = mripy_plot_DCM(dcm, est3, vis);
%   vis.legend.Position = vis.legend.Position + [0.05, 0, 0, 0];

    % dcm
    n_rois = size(dcm.a, 1);
    n_inputs = size(dcm.c, 2);
    assert(size(dcm.b, 1)==n_rois);
    assert(size(dcm.b, 3)==n_inputs);
    assert(size(dcm.c, 1)==n_rois);
    % mods, dris
    mods = find(squeeze(any(any(dcm.b, 1), 2)))'; % Indices for modulatory inputs
    n_mods = sum(mods); % Number of modulatory effects
    dris = find(any(dcm.c, 1)); % Indices for driving inputs
    n_dris = sum(dris); % Number of driving inputs
    % labels
    assert(length(dcm.labels)==n_rois);
    % layout
    assert(all(size(dcm.layout)==[n_rois,2]));
    % node_size
    if ~isfield(dcm, 'node_size')
        dcm.node_size = 10;
    end
    % angle_offset
    if ~isfield(dcm, 'angle_offset')
        dcm.angle_offset = pi/12;
    end
    % font_size
    if ~isfield(dcm, 'font_size')
        dcm.font_size = 18;
    end
    % P_th
    if ~isfield(dcm, 'P_th')
        dcm.P_th = 0.95;
    end
    
    % est
    if nargin < 2
        est = [];
    end
    if ~isempty(est)
        for X = 'ABC'
            if ~isfield(est, X)
                continue;
            end
            if ~isfield(est, [X,'max'])
                est.([X,'max']) = max(abs(est.(X)(:)));
            end
            if ~isfield(est, [X,'min'])
                est.([X,'min']) = 0;
            end
            if ~isfield(est, ['norm',X])
                est.(['norm',X]) = @(x) (x-est.([X,'min']))./(est.([X,'max'])-est.([X,'min']));
            end
        end
    end
    
    % vis
    if nargin < 3
        vis = [];
    end
    if ~isfield(vis, 'mode')
        vis.mode = 'standard';
        % If vis.mode == 'single-modulatory', we also need
        % vis.input: int
    end
    default_vis = [];
    default_vis.conn_colors = {[0.4940, 0.1840, 0.5560], [0.5, 0.5, 0.5]}; % pos|neg
    default_vis.conn_styles = {'-', '-'; '--', '--'}; % sig-pos|sig-neg; insig-pos|insig_neg
    default_input_colors = {
        [0.6350, 0.0780, 0.1840], [0,      0.4470, 0.7410];
        [0.9290, 0.6940, 0.1250], [0.4660, 0.6740, 0.1880];
        [0.8500, 0.3250, 0.0980], [0.3010, 0.7450, 0.9330];
        [1, 0, 0],                [0, 0, 1];
    }; % pos|neg: C3|C0, y|g, o|c, r|b
    default_vis.input_colors = default_input_colors(1:n_inputs,:);
    default_vis.input_styles = {'-', '-'; '--', '--'}; % sig-pos|sig-neg; insig-pos|insig_neg
    default_vis.max_width = 5;
    default_vis.min_width = 0.5;
    default_vis.show_ns_BC = false;
    % conn_colors
    if ~isfield(vis, 'conn_colors')
        switch vis.mode
            case 'single-modulatory'
                vis.conn_colors = {[0.6350, 0.0780, 0.1840], [0, 0.4470, 0.7410]}; % pos|neg
            otherwise
                vis.conn_colors = default_vis.conn_colors;
        end
    end
    if ~isfield(vis, 'conn_styles')
        vis.conn_styles = default_vis.conn_styles;
    end
    % conn_alphas
    if ~isfield(vis, 'conn_alphas')
        switch vis.mode
            case 'single-modulatory'
                vis.conn_alphas = {1, 0.2}; % sig|insig
            otherwise
                vis.conn_alphas = {1, 1};
        end
    end
    % input_colors
    if ~isfield(vis, 'input_colors')
        vis.input_colors = default_vis.input_colors;
    end
    if ~isfield(vis, 'input_styles')
        vis.input_styles = default_vis.input_styles;
    end
    % max_width
    if ~isfield(vis, 'max_width')
        vis.max_width = default_vis.max_width;
    end
    if ~isfield(vis, 'min_width')
        vis.min_width = default_vis.min_width;
    end
    % mod_offset
    if ~isfield(vis, 'mod_offset')
        vis.mod_offset = 0.4;
    end
    % show_ns_BC
    if ~isfield(vis, 'show_ns_BC')
        if strcmpi(vis.mode, 'single-modulatory')
            vis.show_ns_BC = true;
        else
            vis.show_ns_BC = default_vis.show_ns_BC;
        end
    end
    % show_legend
    if ~isfield(vis, 'show_legend')
        vis.show_legend = true;
    end
    if ~isfield(vis, 'legend_values')
        vis.legend_values = [10, 3, 1, -1, -3, -10]; % z
    end
    % conn_values
    if ~isfield(vis, 'conn_values')
        if strcmpi(vis.mode, 'single-modulatory')
            vis.conn_values = est.B(:,:,vis.input);
        else
            vis.conn_values = [];
        end
    end
    if ~isfield(vis, 'conn_label_x')
        vis.conn_label_x = 0.4;
    end
    if ~isfield(vis, 'conn_label_y')
        vis.conn_label_y = 1;
    end
    
    % Plot nodes
    for k = 1:n_rois
        xywh = [dcm.layout(k,:)-dcm.node_size/2, [1,1]*dcm.node_size];
        rectangle('Position', xywh, 'Curvature', [1,1]);
        text(dcm.layout(k,1), dcm.layout(k,2), dcm.labels{k}, ...
            'HorizontalAlignment', 'center', 'FontSize', dcm.font_size, 'FontWeight', 'bold');
    end
    % Plot connections (A)
    arrows = zeros(n_rois,n_rois,6); % [x_start, y_start, x_end, y_end, d, a]
    angles = cell(n_rois,1);
    for m = 1:n_rois
        for n = 1:n_rois
            if m ~= n && dcm.a(m,n) % Non-self-connections from node0 to node1
                r = dcm.node_size/2;
                da = dcm.angle_offset;
                x0 = dcm.layout(n,1);
                x1 = dcm.layout(m,1);
                dx = x1 - x0;
                y0 = dcm.layout(n,2);
                y1 = dcm.layout(m,2);
                dy = y1 - y0;
                a0 = atan2(dy, dx);
                a1 = atan2(-dy, -dx);
                b0 = a0 + da;
                b1 = a1 - da;
                x_start = x0 + r*cos(b0);
                y_start = y0 + r*sin(b0);
                x_end = x1 + r*cos(b1);
                y_end = y1 + r*sin(b1);
                d = sqrt((x_start-x_end)^2 + (y_start-y_end)^2);
                if strcmpi(vis.mode, 'single-modulatory')
                    % pass
                else
                    if isempty(est) || ~isfield(est, 'A')
                        arrow = annotation('arrow');
                    else
                        neg_idx = (est.A(m,n) < 0) + 1;
                        ns_idx = (est.PA(m,n) < dcm.P_th) + 1;
                        arrow = annotation('arrow', ...
                            'Color', vis.conn_colors{1,neg_idx}, ...
                            'LineStyle', vis.conn_styles{ns_idx,neg_idx}, ...
                            'LineWidth', vis.min_width + est.normA(abs(est.A(m,n)))*vis.max_width, ...
                            'HeadWidth', 10 + est.normA(abs(est.A(m,n)))*vis.max_width, ...
                            'HeadStyle', 'cback1');
                    end
                    arrow.Parent = gca();
                    arrow.Position = [x_start, y_start, x_end-x_start, y_end-y_start];
                end
                % Cache geon data
                arrows(m,n,:) = [x_start, y_start, x_end, y_end, d, a0];
                angles{n} = cat(2, angles{n}, b0);
                angles{m} = cat(2, angles{m}, b1);
            elseif m == n && (~isempty(est) && isfield(est, 'A'))
                if strcmpi(vis.mode, 'single-modulatory')
                    % pass
                else
                    pos_idx = (est.A(m,n) > 0) + 1;
                    ns_idx = (est.PA(m,n) < dcm.P_th) + 1;
                    xywh = [dcm.layout(m,:)-dcm.node_size/2, [1,1]*dcm.node_size];
                    rectangle('Position', xywh, 'Curvature', [1,1], ...
                        'EdgeColor', vis.conn_colors{1,pos_idx}, ...
                        'LineStyle', vis.conn_styles{ns_idx,pos_idx}, ...
                        'LineWidth', vis.min_width + est.normA(abs(est.A(m,n)))*vis.max_width);
                end
            end
        end
    end
    % Plot modulatory (B)
    s = 1;
    L_total = 2*(n_mods-1)*s;
    L_deltas = linspace(0, L_total, 2*n_mods-1) - (n_mods-1)*s;
    kk = 0;
    for k = mods
        kk = kk + 1;
        for m = 1:n_rois
            for n = 1:n_rois
                if m ~= n && dcm.b(m,n,k)
                    xc = vis.mod_offset*arrows(m,n,1) + (1-vis.mod_offset)*arrows(m,n,3);
                    yc = vis.mod_offset*arrows(m,n,2) + (1-vis.mod_offset)*arrows(m,n,4);
                    x = xc + L_deltas(kk)*cos(arrows(m,n,6));
                    y = yc + L_deltas(kk)*sin(arrows(m,n,6));
                    if isempty(est) || ~isfield(est, 'B')
                        rectangle('Position',[x-s/2, y-s/2, s, s], 'Curvature', [1,1], ...
                            'FaceColor', vis.input_colors{k,1}, 'EdgeColor', 'w');
                    else
                        neg_idx = (est.B(m,n,k) < 0) + 1;
                        ns_idx = (est.PB(m,n,k) < dcm.P_th) + 1;
                        if strcmpi(vis.mode, 'single-modulatory')
                            if k == vis.input
                                if ns_idx == 2 && ~vis.show_ns_BC % insignificant
                                    continue;
                                end
                                arrow = annotation('arrow', ...
                                    'Color', (vis.conn_colors{1,neg_idx}-1)*vis.conn_alphas{1,ns_idx}+1, ...
                                    'LineStyle', vis.conn_styles{ns_idx,neg_idx}, ...
                                    'LineWidth', vis.min_width + est.normB(abs(est.B(m,n,k)))*vis.max_width, ...
                                    'HeadWidth', 10 + est.normB(abs(est.B(m,n,k)))*vis.max_width, ...
                                    'HeadStyle', 'cback1');
                                arrow.Parent = gca();
                                arrow.Position = [arrows(m,n,1), arrows(m,n,2), arrows(m,n,3)-arrows(m,n,1), arrows(m,n,4)-arrows(m,n,2)];
                                % Values
                                a = arrows(m,n,6)+pi/2;
                                x = vis.conn_label_x*arrows(m,n,1)+(1-vis.conn_label_x)*arrows(m,n,3) + cos(a)*vis.conn_label_y;
                                y = vis.conn_label_x*arrows(m,n,2)+(1-vis.conn_label_x)*arrows(m,n,4) + sin(a)*vis.conn_label_y;
                                if ns_idx == 2 % insignificant
                                    c = [1, 1, 1]*(1-vis.conn_alphas{1,2});
                                else
                                    c = vis.conn_colors{1,neg_idx};
                                end
                                h = text(x, y, sprintf('%.2f', vis.conn_values(m,n)), ...
                                    'HorizontalAlignment', 'center', ...
                                    'Color', c);
                                h.Rotation = -mod(-arrows(m,n,6)*180/pi-90,180)+90;
                            end
                        else
                            if ns_idx == 1 % significant
                                rectangle('Curvature', [1,1], ...
                                    'Position',[x,y,0,0]+[-1/2,-1/2,1,1]*s*(1+est.normB(abs(est.B(m,n,k)))), ...
                                    'FaceColor', vis.input_colors{k,neg_idx}, 'EdgeColor', 'w');
                            elseif vis.show_ns_BC
                                rectangle('Curvature', [1,1], ...
                                    'Position',[x,y,0,0]+[-1/2,-1/2,1,1]*s*(1+est.normB(abs(est.B(m,n,k)))), ...
                                    'FaceColor', 'w', 'EdgeColor', vis.input_colors{k,neg_idx});
                            end
                        end
                    end
                else % m == n
                end
            end
        end
    end
    % Plot driving (C)
    r = dcm.node_size/2;
    da = dcm.angle_offset;
    a_deltas = ((0:n_dris-1)-(n_dris-1)/2)*da;
    kk = 0;
    arrow = [];
    for k = dris
        kk = kk + 1;
        if strcmpi(vis.mode, 'single-modulatory')
            continue;
        end
        for m = 1:n_rois
            if dcm.c(m,k)
                a = angle(mean(exp(1j*(angles{m}+pi))));
                b = a + a_deltas(kk);
                x_end = dcm.layout(m,1) + r*cos(b);
                y_end = dcm.layout(m,2) + r*sin(b);
                x_start = x_end + r*cos(a);
                y_start = y_end + r*sin(a);
                if isempty(est) || ~isfield(est, 'C') % Without est
                    arrow = annotation('arrow', 'Color', vis.input_colors{k,1}, 'LineWidth', 2);
                else % With est
                    neg_idx = (est.C(m,k) < 0) + 1;
                    ns_idx = (est.PC(m,k) < dcm.P_th) + 1;
                    if ns_idx == 1 || vis.show_ns_BC
                        arrow = annotation('arrow', 'LineWidth', 2, ...
                            'Color', vis.input_colors{k,neg_idx}, ...
                            'LineStyle', vis.input_styles{ns_idx,neg_idx}, ...
                            'LineWidth', est.normC(abs(est.C(m,k)))*vis.max_width+vis.min_width, ...
                            'HeadWidth', 10+2*est.normC(abs(est.C(m,k)))*vis.max_width, ...
                            'HeadStyle', 'cback1');
                    end
                end
                if ~isempty(arrow)
                    arrow.Parent = gca();
                    arrow.Position = [x_start, y_start, x_end-x_start, y_end-y_start];
                    arrow = [];
                end
            end
        end
    end
    % Legend
    if strcmpi(vis.mode, 'single-modulatory') && vis.show_legend
        hold on;
        zs = vis.legend_values;
        for k = 1:length(zs)
            lines(k) = plot([0,0], [0,0], 'Visible', 'on', ...
                'Color', vis.conn_colors{1,(zs(k)<0)+1}, ...
                'LineWidth', vis.min_width + est.normB(abs(zs(k)))*vis.max_width);
            labels{k} = ['\color{black}', num2str(zs(k))];
        end
        vis.legend = legend(lines, labels, 'Location', 'east', 'Fontsize', 12);
        legend('boxoff');
        title(vis.legend, 'z');
    end
    % Finalize
    xlim_ = xlim();
    xlim(xlim_ + [-1,1]*diff(xlim_)*0.1);
    ylim_ = ylim();
    ylim(ylim_ + [-1,1]*diff(ylim_)*0.1);
    axis('equal');
    set(gca, 'XTick', [], 'YTick', []); % axis('off');
end

