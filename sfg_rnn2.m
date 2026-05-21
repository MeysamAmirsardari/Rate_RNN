clear; close all; clc;
rng(7);


b = [1 sqrt(3) 0 -sqrt(3) -1].' ;
H = fft(b,1024);
f = (0 : 1/1024 : 1-1/1024).';
A = abs(H);
plot(f,A)
q = angle(H);
% plot(f,q)

%% =========================================================================
%  1.  NETWORK & SIMULATION PARAMETERS
% =========================================================================
N      = 12;
N_fig  = 4;
N_gnd  = N - N_fig;

USE_RANDOM_FIG_IDX = false;
if USE_RANDOM_FIG_IDX
    fig_idx = sort(randperm(N, N_fig));
else
    fig_idx = [2, 5, 8, 11];
end
gnd_idx = setdiff(1:N, fig_idx);

T  = 5.5;
dt = 0.001;
nT = round(T/dt);
t_vec = (0:nT-1)*dt;

pulse_dur    = 0.025;
pulse_per    = 0.250;
pulse_steps  = round(pulse_dur/dt);
period_steps = round(pulse_per/dt);
stim_amp     = 50;

FIG_RANDOM_EXTRA   = true;
n_fig_extra_pulses = 15;


%% =========================================================================
%  2.  STYLE PALETTE
% =========================================================================
% Figure / Ground — green & blue matching the SFG paradigm reference figure
COL_FIG   = [0.30 0.78 0.35];          % figure  : saturated leaf green
COL_GND   = [0.30 0.46 0.75];          % ground  : medium blue
COL_FG    = [0.45 0.45 0.45];          % cross   : mid gray

% Excitation / Inhibition palette for figure 3
COL_EXC   = [0.95 0.50 0.15];          % excitation : orange
COL_INH   = [0.20 0.40 0.75];          % inhibition : blue
COL_EXC_LIGHT = 'black'; %COL_EXC*0.5 + 0.5;
COL_INH_LIGHT =  COL_INH*0.5 + 0.5;

COL_AXIS     = [0.20 0.20 0.20];
COL_PANEL    = [0.985 0.985 0.99];
COL_FIG_BAND = 1 - 0.07*(1 - COL_FIG); % very faint green row bands
FONT_NAME    = 'Helvetica';

% TeX color directive for green figure-channel tick labels
TEX_GREEN = sprintf('\\color[rgb]{%.3f,%.3f,%.3f}', ...
                    COL_FIG(1), COL_FIG(2), COL_FIG(3));


%% =========================================================================
%  3.  TM / RATE / INHIBITION / PLASTICITY PARAMETERS
% =========================================================================
tau_D = 0.200;  tau_F = 0.500;  U_base = 0.03;  A_scale = 15.0;
tau_E = 0.020;  tau_I = 0.080;  phi = @(z) max(z,0);
W_EI_local = 1.0;  W_IE_global = 1.0;
tau_trace = 0.040;
eta       = 1.5;
eta_LTD   = 0.8;
decay_W   = 0.05;
W_max     = 2.0;
W_norm    = 50.0;


%% =========================================================================
%  4.  STIMULUS  ( coherent figure + sparse extras + random ground )
% =========================================================================
stim = zeros(N, nT);

fig_onsets = 1:period_steps:(nT-pulse_steps+1);
n_pulses   = numel(fig_onsets);
for k = 1:n_pulses
    idx = fig_onsets(k) : (fig_onsets(k) + pulse_steps - 1);
    stim(fig_idx, idx) = stim_amp;
end

if FIG_RANDOM_EXTRA && n_fig_extra_pulses > 0
    for ci = 1:numel(fig_idx)
        ch       = fig_idx(ci);
        occupied = stim(ch, :) > 0;
        placed   = 0; tries = 0;
        while placed < n_fig_extra_pulses && tries < n_fig_extra_pulses*500
            on  = randi(nT - pulse_steps + 1);
            idx = on : (on + pulse_steps - 1);
            if ~any(occupied(idx))
                occupied(idx) = true;
                stim(ch, idx) = stim_amp;
                placed = placed + 1;
            end
            tries = tries + 1;
        end
    end
end

for ci = 1:numel(gnd_idx)
    ch       = gnd_idx(ci);
    occupied = false(1, nT);
    placed   = 0; tries = 0;
    while placed < n_pulses && tries < n_pulses*500
        on  = randi(nT - pulse_steps + 1);
        idx = on : (on + pulse_steps - 1);
        if ~any(occupied(idx))
            occupied(idx) = true;
            stim(ch, idx) = stim_amp;
            placed = placed + 1;
        end
        tries = tries + 1;
    end
end


%% =========================================================================
%  5.  STATE INITIALISATION
% =========================================================================
E   = zeros(N,1);
Iv  = zeros(N,1);
u   = U_base * ones(N,1);
x   = ones(N,1);
Tr  = zeros(N,1);
W   = zeros(N,N);

E_hist      = zeros(N, nT);
I_hist      = zeros(N, nT);
W_FF        = zeros(1, nT);
W_GG        = zeros(1, nT);
W_FG        = zeros(1, nT);
Exc_in_hist = zeros(N, nT);
glob_I_hist = zeros(1, nT);

off_diag = ~eye(N);
mask_FF  = false(N); mask_FF(fig_idx, fig_idx) = true; mask_FF = mask_FF & off_diag;
mask_GG  = false(N); mask_GG(gnd_idx, gnd_idx) = true; mask_GG = mask_GG & off_diag;
mask_FG  = false(N);
mask_FG(fig_idx, gnd_idx) = true;
mask_FG(gnd_idx, fig_idx) = true;
mask_FG  = mask_FG & off_diag;


%% =========================================================================
%  6.  MAIN INTEGRATION LOOP
% =========================================================================
for t = 1:nT
    s = stim(:, t);
    tm_in   = A_scale .* u .* x .* s;
    rec_E   = W * E;
    glob_I  = W_IE_global * sum(Iv);
    net_E   = tm_in + rec_E - glob_I;
    net_I   = W_EI_local * E;

    dE  = (-E  + phi(net_E)) / tau_E;
    dI  = (-Iv + phi(net_I)) / tau_I;
    du  = (U_base - u)/tau_F + U_base .* (1 - u) .* s;
    dx  = (1 - x)/tau_D       - u .* x .* s;
    dTr = (-Tr + E) / tau_trace;

    En  = E  / W_norm;
    Trn = Tr / W_norm;
    dW  = eta * (En * Trn.') - eta_LTD * (Trn * En.') - decay_W * W;

    E  = E  + dt*dE;
    Iv = Iv + dt*dI;
    u  = u  + dt*du;
    x  = x  + dt*dx;
    Tr = Tr + dt*dTr;
    W  = W  + dt*dW;

    W(1:N+1:end) = 0;
    W = min(max(W,0), W_max);

    E_hist(:,t)      = E;
    I_hist(:,t)      = Iv;
    Exc_in_hist(:,t) = tm_in + rec_E;
    glob_I_hist(t)   = glob_I;
    W_FF(t)          = mean(W(mask_FF));
    W_GG(t)          = mean(W(mask_GG));
    W_FG(t)          = mean(W(mask_FG));
end
W_final = W;


%% =========================================================================
%  7.  FIGURE 1  :  stimulus raster  +  E activity  +  weight evolution
%      Explicit subplot positions -> guaranteed gaps, no title overlap.
% =========================================================================
fig1 = figure('Position',[80 30 1200 1200],'Color','w');

% --- Layout (normalised units) ---
fig1_left   = 0.08;  fig1_right  = 0.04;
fig1_w      = 1 - fig1_left - fig1_right;
fig1_bottom = 0.06;  fig1_top    = 0.08;       % room for sgtitle
fig1_gap    = 0.11;                            % vertical gap between panels
fig1_pan_h  = 1 - fig1_top - fig1_bottom - 2*fig1_gap;
h1 = 0.45 * fig1_pan_h;        % stimulus raster : 50%
h2 = 0.25 * fig1_pan_h;        % activity        : 25%
h3 = 0.25 * fig1_pan_h;        % weights         : 25%
y3 = fig1_bottom;
y2 = y3 + h3 + fig1_gap;
y1 = y2 + h2 + fig1_gap;

% ---- (1) Stimulus raster  -  patch rectangles per pulse ---------------
ax1 = subplot('Position', [fig1_left, y1, fig1_w, h1]);
hold on;

% Subtle pale-green row bands for figure channels (groups non-contiguous ch.)
for ci = 1:numel(fig_idx)
    ch = fig_idx(ci);
    rectangle('Position', [-1, ch-0.5, T+2, 1], ...
              'FaceColor', COL_FIG_BAND, 'EdgeColor', 'none');
end

% Draw each pulse as a rectangle of *exact* width = pulse_dur (25 ms in data).
% Pulses for one channel go in a single PATCH call for speed and cleanness.
% Draw each pulse as a rectangle of *exact* width = pulse_dur (25 ms in data).
bar_h = 0.7;
for ch = 1:N
    on_starts = find(diff([0, double(stim(ch,:) > 0)]) == 1);
    n_pls = numel(on_starts);
    if n_pls == 0, continue; end
    
   if any(ch == fig_idx)
        is_coh = ismember(on_starts, fig_onsets);
    else
        is_coh = false(1, n_pls);
end
    
   coh_starts = on_starts(is_coh);
    if ~isempty(coh_starts)
        x_left  = (coh_starts - 1) * dt;
        x_right = x_left + pulse_dur;
        X_coh = [x_left; x_right; x_right; x_left];
        Y_coh = [ch-bar_h/2; ch-bar_h/2; ch+bar_h/2; ch+bar_h/2] * ones(1, numel(coh_starts));
        patch(X_coh, Y_coh, COL_FIG, 'EdgeColor', 'none');
    end
    
   
    rnd_starts = on_starts(~is_coh);
    if ~isempty(rnd_starts)
        x_left  = (rnd_starts - 1) * dt;
        x_right = x_left + pulse_dur;
        X_rnd = [x_left; x_right; x_right; x_left];
        Y_rnd = [ch-bar_h/2; ch-bar_h/2; ch+bar_h/2; ch+bar_h/2] * ones(1, numel(rnd_starts));
        patch(X_rnd, Y_rnd, COL_GND, 'EdgeColor', 'none');
    end
end

xlim([-0.05 T+0.05]); ylim([0.4 N+0.6]);

% Build TeX-coloured y-tick labels : green for figure channels, black otherwise
ytick_labels = cell(1, N);
for ch = 1:N
    if any(ch == fig_idx)
        ytick_labels{ch} = sprintf('%s\\bf %d', TEX_GREEN, ch);
    else
        ytick_labels{ch} = sprintf('%d', ch);
    end
end

set(gca, 'YDir', 'reverse', 'YTick', 1:N, ...
         'YTickLabel', ytick_labels, 'TickLabelInterpreter', 'tex', ...
         'FontSize', 12, 'FontName', FONT_NAME, ...
         'TickDir', 'out', 'LineWidth', 1, 'Color', COL_PANEL, ...
         'XColor', COL_AXIS, 'YColor', COL_AXIS, 'Layer', 'top');
xlabel('Time (s)',          'FontSize', 13, 'FontName', FONT_NAME);
ylabel('Frequency channel', 'FontSize', 13, 'FontName', FONT_NAME);
title('Stimulus raster   (figure channels in green, ground channels in blue)', ...
      'FontWeight','bold','FontSize',13,'FontName',FONT_NAME);
box on;

% ---- (2) Mean E activity ----------------------------------------------
ax2 = subplot('Position', [fig1_left, y2, fig1_w, h2]);
plot(t_vec, mean(E_hist(gnd_idx,:),1), 'Color', COL_GND, 'LineWidth', 2.5); hold on;
plot(t_vec, mean(E_hist(fig_idx,:),1), 'Color', COL_FIG, 'LineWidth', 2.5);
xlabel('Time (s)',    'FontSize', 12, 'FontName', FONT_NAME);
ylabel('Mean E rate', 'FontSize', 12, 'FontName', FONT_NAME);
title('Mean excitatory activity (group average)', ...
      'FontWeight','bold','FontSize',12,'FontName',FONT_NAME);
lg = legend({'Ground','Figure'}, 'Location','northeast');
set(lg,'Box','off','FontSize',11);
grid on; box off;
set(gca, 'FontSize',11,'FontName',FONT_NAME,'TickDir','out', ...
         'LineWidth',1,'GridAlpha',0.2, ...
         'XColor',COL_AXIS,'YColor',COL_AXIS);

% ---- (3) Mean E->E weight evolution -----------------------------------
ax3 = subplot('Position', [fig1_left, y3, fig1_w, h3]);
plot(t_vec, W_FF, 'Color', COL_FIG, 'LineWidth', 2.5); hold on;
plot(t_vec, W_GG, 'Color', COL_GND, 'LineWidth', 2.5);
plot(t_vec, W_FG, 'Color', COL_FG,  'LineWidth', 1.9, 'LineStyle', '--');
xlabel('Time (s)',             'FontSize', 12, 'FontName', FONT_NAME);
ylabel('Mean synaptic weight', 'FontSize', 12, 'FontName', FONT_NAME);
title('Recurrent E\rightarrowE weight evolution', ...
      'FontWeight','bold','FontSize',12,'FontName',FONT_NAME);
lg = legend({'Fig \rightarrow Fig','Gnd \rightarrow Gnd','Fig \leftrightarrow Gnd'}, ...
            'Location','northwest');
set(lg,'Box','off','FontSize',11);
grid on; box off;
set(gca, 'FontSize',11,'FontName',FONT_NAME,'TickDir','out', ...
         'LineWidth',1,'GridAlpha',0.2, ...
         'XColor',COL_AXIS,'YColor',COL_AXIS);
ymax_w = max([W_FF, W_GG, W_FG]) * 1.15;
ylim([0 max(ymax_w, 0.02)]);

% try
%     sgtitle('Stochastic Figure-Ground   -   stimulus, activity, plasticity', ...
%             'FontWeight','bold','FontSize',14,'FontName',FONT_NAME);
% catch
% end

print(fig1, 'SFG_simulation_results.png','-dpng','-r150');
fprintf('Figure 1 saved to SFG_simulation_results.png\n');


%% =========================================================================
%  8.  FIGURE 2  :  CONNECTIVITY MAP  -  two views (side by side)
% =========================================================================
fig2 = figure('Position',[100 200 1500 720],'Color','w');

% --- Layout
fig2_left = 0.05; fig2_right = 0.03;
fig2_bottom = 0.12; fig2_top = 0.12;
fig2_gap_h  = 0.10;
fig2_w_each = (1 - fig2_left - fig2_right - fig2_gap_h) / 2;
fig2_h = 1 - fig2_top - fig2_bottom;

% Shared color scale (so the two heatmaps are directly comparable)
clim_v = [0, max(W_final(:)) * 1.02];

% Build TeX tick label cells
% -- original-order labels
ticklab_orig = cell(1, N);
for ch = 1:N
    if any(ch == fig_idx)
        ticklab_orig{ch} = sprintf('%s\\bf %d', TEX_GREEN, ch);
    else
        ticklab_orig{ch} = sprintf('%d', ch);
    end
end
% -- reordered labels (perm = [fig_idx, gnd_idx], display original numbers)
perm = [fig_idx, gnd_idx];
ticklab_perm = cell(1, N);
for k = 1:N
    ch = perm(k);
    if k <= numel(fig_idx)
        ticklab_perm{k} = sprintf('%s\\bf %d', TEX_GREEN, ch);
    else
        ticklab_perm{k} = sprintf('%d', ch);
    end
end

% ---- (a) REORDERED  -  figure channels grouped at top-left ------------
ax_a = subplot('Position', [fig2_left, fig2_bottom, fig2_w_each, fig2_h]);
W_perm = W_final(perm, perm);
imagesc(W_perm);
try, colormap(ax_a, parula); catch, colormap(ax_a, hot); end
caxis(clim_v);
cb_a = colorbar; ylabel(cb_a, 'Weight', 'FontSize', 11, 'FontName', FONT_NAME);
hold on;
rectangle('Position', [0.5, 0.5, numel(fig_idx), numel(fig_idx)], ...
          'EdgeColor', COL_FIG, 'LineWidth', 3);
hold off;
axis equal tight;
set(gca, 'XTick', 1:N, 'YTick', 1:N, ...
         'XTickLabel', ticklab_perm, 'YTickLabel', ticklab_perm, ...
         'TickLabelInterpreter', 'tex', ...
         'FontSize', 11, 'FontName', FONT_NAME, 'TickDir', 'out', ...
         'LineWidth', 1, 'XColor', COL_AXIS, 'YColor', COL_AXIS, ...
         'Layer', 'top');
xlabel('Pre-synaptic channel',  'FontSize', 12, 'FontName', FONT_NAME);
ylabel('Post-synaptic channel', 'FontSize', 12, 'FontName', FONT_NAME);
title('Reordered : figure channels grouped (top-left)', ...
      'FontWeight','bold','FontSize',13,'FontName',FONT_NAME);

% ---- (b) ORIGINAL channel order ---------------------------------------
ax_b = subplot('Position', ...
               [fig2_left + fig2_w_each + fig2_gap_h, fig2_bottom, fig2_w_each, fig2_h]);
imagesc(W_final);
try, colormap(ax_b, parula); catch, colormap(ax_b, hot); end
caxis(clim_v);
cb_b = colorbar; ylabel(cb_b, 'Weight', 'FontSize', 11, 'FontName', FONT_NAME);
hold on;
for ci = 1:numel(fig_idx)
    for cj = 1:numel(fig_idx)
        if fig_idx(ci) ~= fig_idx(cj)
            rectangle('Position', [fig_idx(cj)-0.5, fig_idx(ci)-0.5, 1, 1], ...
                      'EdgeColor', COL_FIG, 'LineWidth', 2.5);
        end
    end
end
hold off;
axis equal tight;
set(gca, 'XTick', 1:N, 'YTick', 1:N, ...
         'XTickLabel', ticklab_orig, 'YTickLabel', ticklab_orig, ...
         'TickLabelInterpreter', 'tex', ...
         'FontSize', 11, 'FontName', FONT_NAME, 'TickDir', 'out', ...
         'LineWidth', 1, 'XColor', COL_AXIS, 'YColor', COL_AXIS, ...
         'Layer', 'top');
xlabel('Pre-synaptic channel',  'FontSize', 12, 'FontName', FONT_NAME);
ylabel('Post-synaptic channel', 'FontSize', 12, 'FontName', FONT_NAME);
title('Original channel order', ...
      'FontWeight','bold','FontSize',13,'FontName',FONT_NAME);

% try
%     sgtitle('Final learned weight matrix W_{ij}    (green: figure-figure submatrix)', ...
%             'FontWeight','bold','FontSize',14,'FontName',FONT_NAME);
% catch
% end

print(fig2, 'SFG_connectivity_map.png','-dpng','-r150');
fprintf('Figure 2 saved to SFG_connectivity_map.png\n');


%% =========================================================================
%  9.  FIGURE 3  :  synaptic inputs  -  orange (exc) / blue (inh) palette
% =========================================================================
Exc_fig = mean(Exc_in_hist(fig_idx, :), 1);
Exc_gnd = mean(Exc_in_hist(gnd_idx, :), 1);
Inh_fig = glob_I_hist;
Inh_gnd = glob_I_hist;
Net_fig = Exc_fig - Inh_fig;
Net_gnd = Exc_gnd - Inh_gnd;

fig3 = figure('Position',[120 60 1200 1100],'Color','w');

% --- Layout: 3 panels with explicit positions -> no title overlap ---
fig3_left   = 0.08;  fig3_right  = 0.04;
fig3_w      = 1 - fig3_left - fig3_right;
fig3_bottom = 0.06;  fig3_top    = 0.08;
fig3_gap    = 0.10;
fig3_pan_h  = (1 - fig3_top - fig3_bottom - 2*fig3_gap) / 3;
y3_3 = fig3_bottom;
y3_2 = y3_3 + fig3_pan_h + fig3_gap;
y3_1 = y3_2 + fig3_pan_h + fig3_gap;

% ---- (1) Inhibition  -  all BLUE --------------------------------------
ax_i = subplot('Position', [fig3_left, y3_1, fig3_w, fig3_pan_h]);
plot(t_vec, Inh_gnd, 'Color', 'black',       'LineWidth', 2.2); hold on;
plot(t_vec, Inh_fig, 'Color', COL_INH_LIGHT, 'LineWidth', 1.2, 'LineStyle','--');
xlabel('Time (s)',           'FontSize', 12, 'FontName', FONT_NAME);
ylabel('Inhibitory current', 'FontSize', 12, 'FontName', FONT_NAME);
title('Inhibition received   (global  -  figure and ground coincide)', ...
      'FontWeight','bold','FontSize',12,'FontName',FONT_NAME);
lg = legend({'Ground','Figure'},'Location','northeast');
set(lg,'Box','off','FontSize',11);
grid on; box off;
set(gca,'FontSize',11,'FontName',FONT_NAME,'TickDir','out', ...
        'LineWidth',1,'GridAlpha',0.2, ...
        'XColor',COL_AXIS,'YColor',COL_AXIS);

% ---- (2) Excitation  -  all ORANGE ------------------------------------
ax_e = subplot('Position', [fig3_left, y3_2, fig3_w, fig3_pan_h]);
hold on;
plot(t_vec, Exc_fig, 'Color', COL_EXC,       'LineWidth', 1.7);
plot(t_vec, Exc_gnd, 'Color', COL_EXC_LIGHT, 'LineWidth', 1.9); 

xlabel('Time (s)',           'FontSize', 12, 'FontName', FONT_NAME);
ylabel('Excitatory current', 'FontSize', 12, 'FontName', FONT_NAME);
title('Excitation received   (feed-forward input  +  recurrent W*E)', ...
      'FontWeight','bold','FontSize',12,'FontName',FONT_NAME);
lg = legend({'Ground','Figure'},'Location','northeast');
set(lg,'Box','off','FontSize',11);
grid on; box off;
set(gca,'FontSize',11,'FontName',FONT_NAME,'TickDir','out', ...
        'LineWidth',1,'GridAlpha',0.2, ...
        'XColor',COL_AXIS,'YColor',COL_AXIS);

% ---- (3) Net drive  -  fig in ORANGE (E-dominant), gnd in BLUE (I-dom) -
ax_n = subplot('Position', [fig3_left, y3_3, fig3_w, fig3_pan_h]);
 hold on;
plot(t_vec, Net_fig, 'Color', COL_EXC, 'LineWidth', 1.6);
plot(t_vec, Net_gnd, 'Color', COL_INH, 'LineWidth', 1.8);
plot(xlim, [0 0], ':', 'Color', [0.2 0.2 0.2], 'LineWidth', 0.8);
xlabel('Time (s)',           'FontSize', 12, 'FontName', FONT_NAME);
ylabel('Net synaptic drive', 'FontSize', 12, 'FontName', FONT_NAME);
title('Net synaptic drive  =  excitation  -  inhibition   (input to E pool)', ...
      'FontWeight','bold','FontSize',12,'FontName',FONT_NAME);
lg = legend({'Ground','Figure'},'Location','northeast');
set(lg,'Box','off','FontSize',11);
grid on; box off;
set(gca,'FontSize',11,'FontName',FONT_NAME,'TickDir','out', ...
        'LineWidth',1,'GridAlpha',0.2, ...
        'XColor',COL_AXIS,'YColor',COL_AXIS);

% try
%     sgtitle('Synaptic inputs to figure vs ground E pools', ...
%             'FontWeight','bold','FontSize',14,'FontName',FONT_NAME);
% catch
% end

print(fig3, 'SFG_inputs_figure.png','-dpng','-r150');
fprintf('Figure 3 saved to SFG_inputs_figure.png\n');


%% =========================================================================
%  10.  SUMMARY STATISTICS
% =========================================================================
fprintf('\n--- Final weight statistics ---------------------------------\n');
fprintf('  Figure channels       : [%s]\n', num2str(fig_idx));
fprintf('  Ground channels       : [%s]\n', num2str(gnd_idx));
fprintf('  Mean Fig-Fig  weight  : %.4f\n', mean(W_final(mask_FF)));
fprintf('  Mean Gnd-Gnd  weight  : %.4f\n', mean(W_final(mask_GG)));
fprintf('  Mean Fig-Gnd  weight  : %.4f\n', mean(W_final(mask_FG)));
fprintf('  Ratio FF / FG         : %.2fx\n', ...
        mean(W_final(mask_FF)) / max(mean(W_final(mask_FG)),1e-9));
fprintf('-------------------------------------------------------------\n');