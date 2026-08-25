function out = sfg_staircase(varargin)
% Stochastic figure-ground with a staircase figure.
%
% Everything is set in milliseconds. Internally time runs on a grid of hopMs
% slots and a tone lasts several of them; if the same number of tones starts
% in every slot, the number sounding is constant and the envelope is uniform
% by construction. hopMs is therefore the resolution of every timing control,
% and it is independent of the tone length.
%
%   sfg_staircase                        % 7-tone staircase, 35 ms per step
%   sfg_staircase('stepMs', 0)           % the classic coherent figure
%   sfg_staircase('stepMs', 10, 'hopMs', 5)
%
% Writes the mix and a figure-only wav, and returns audioplayer objects:
%   out = sfg_staircase; pause(out.player); resume(out.player)
%   play(out.figurePlayer)               % the figure without the cloud

cfg = config();
for i = 1:2:numel(varargin)
    cfg.(varargin{i}) = varargin{i+1};
end
rng(cfg.seed);

pool = make_pool(cfg);
[chan, slot, isFig, k, starts] = schedule(cfg, pool);
out = render(cfg, pool, chan, slot, isFig, k);

report(cfg, pool, chan, slot, isFig, out, k, starts);
if cfg.doPlot, show(cfg, pool, chan, slot, isFig); end

if ~isempty(cfg.wavFile)
    [d, f, e] = fileparts(cfg.wavFile);
    audiowrite(cfg.wavFile, out.y(:), cfg.fs, 'BitsPerSample', 24);
    out.figureFile = fullfile(d, [f '_figure' e]);
    audiowrite(out.figureFile, out.figure(:), cfg.fs, 'BitsPerSample', 24);
    fprintf('  wrote %s and %s\n', cfg.wavFile, out.figureFile);
end

out.player       = audioplayer(out.y, cfg.fs);
out.figurePlayer = audioplayer(out.figure, cfg.fs);
if cfg.doPlay, play(out.player); end
fprintf('  play(out.player) | pause | resume | stop  (also out.figurePlayer)\n');
end

% ------------------------------------------------------------------ config
function cfg = config()
cfg.fs           = 48000;
cfg.seed         = 3;

cfg.hopMs        = 5;       % time grid: the resolution of every control below
cfg.toneMs       = 35;      % tone duration, a whole number of hops

cfg.nTones       = 7;       % tones in the figure
cfg.stepMs       = 35;      % staircase step (0 = coherent chord)
cfg.order        = 'rise';  % 'rise' or 'fall'
cfg.spanSt       = 24;      % frequency span of the figure, semitones
cfg.rateHz       = 5;       % figure repetition rate
cfg.jitterMs     = 40;      % random displacement of each figure onset
cfg.wobbleMs     = 0;       % frozen irregularity of the staircase

cfg.startsPerSlot= 0;       % tones starting per slot; 0 solves for it.
                            % tones sounding = startsPerSlot * toneMs/hopMs,
                            % so a fine hopMs buys resolution with density.
                            % Comparing conditions? Fix this by hand at the
                            % largest any of them needs, or the background
                            % density moves with the step.
cfg.contrast     = 4;       % figure/background per-channel rate, sets the pool
cfg.shareChannels= true;    % let the background use the figure's channels

cfg.fRefHz       = 1000;
cfg.poolSt       = [-24 36];

cfg.durationS    = 20;
cfg.peakDbfs     = -3;
cfg.doPlay       = true;
cfg.doPlot       = true;
cfg.wavFile      = 'sfg_staircase.wav';   % _figure.wav written alongside
end

% -------------------------------------------------------------------- pool
function pool = make_pool(cfg)
% A figure channel sounds rateHz times a second; the background shares the
% rest between the remaining channels, so asking for a contrast fixes the
% pool size.
k = round(cfg.toneMs / cfg.hopMs);
starts = max(cfg.startsPerSlot, 1);
cloudPerS = starts * 1000/cfg.hopMs - cfg.nTones * cfg.rateHz;
want = max(cloudPerS, 1) * cfg.contrast / cfg.rateHz + cfg.nTones;

grids  = 0.25:0.25:12;
counts = floor(diff(cfg.poolSt) ./ grids) + 1;
[~, i] = min(abs(counts - want));

pool.gridSt = grids(i);
pool.st = cfg.poolSt(1):pool.gridSt:cfg.poolSt(2);
pool.f  = cfg.fRefHz * 2.^(pool.st / 12);
pool.n  = numel(pool.st);

lo   = round(0.15 * (pool.n - 1)) + 1;
span = round(cfg.spanSt / pool.gridSt);
if lo + span > pool.n
    error('figure spans %g st, pool only %g st', cfg.spanSt, diff(cfg.poolSt));
end
pool.figIdx = round(linspace(lo, lo + span, cfg.nTones));
if pool.n < k * starts + 2
    error('pool of %d channels cannot hold %d tones sounding at once', ...
        pool.n, k * starts);
end
end

% ---------------------------------------------------------------- schedule
function [chan, slot, isFig, k, starts] = schedule(cfg, pool)
k = cfg.toneMs / cfg.hopMs;
if abs(k - round(k)) > 1e-9
    error('toneMs %g must be a whole number of hopMs %g', ...
        cfg.toneMs, cfg.hopMs);
end
k = round(k);

nSlots = floor(cfg.durationS * 1000 / cfg.hopMs);
period = max(1, round(1000 / (cfg.rateHz * cfg.hopMs)));

lag = round((0:cfg.nTones-1) * cfg.stepMs / cfg.hopMs);
if strcmp(cfg.order, 'fall'), lag = fliplr(lag); end
if cfg.wobbleMs > 0
    lag = lag + randi([0 round(cfg.wobbleMs/cfg.hopMs)], 1, cfg.nTones);
end
jit = round(cfg.jitterMs / cfg.hopMs);

occ = cell(1, nSlots);
for w = 0 : floor(nSlots / period)
    c0 = w * period + 1;
    if jit > 0, c0 = c0 + randi([-jit jit]); end
    for i = 1:cfg.nTones
        c = c0 + lag(i);
        if c >= 1 && c <= nSlots
            occ{c}(end+1) = pool.figIdx(i);
        end
    end
end

% The figure can put several tones in one slot once its copies overlap or the
% jitter shifts them together. That sets the floor on how many tones start
% per slot, and everything sounding follows from it.
most = max(cellfun(@numel, occ));
starts = cfg.startsPerSlot;
if starts == 0
    starts = most;
elseif starts < most
    error(['the figure starts %d tones in one slot, so startsPerSlot must ' ...
           'be at least %d (or reduce jitterMs / stepMs)'], most, most);
end

chan  = zeros(nSlots * starts, 1);
slot  = zeros(size(chan));
isFig = false(size(chan));

pack = randperm(pool.n); p = 1;
live = [];                     % channels still sounding, so never reused
m = 0;
for c = 1:nSlots
    here = occ{c};
    nFig = numel(here);
    tries = 0;
    while numel(here) < starts
        if p > numel(pack), pack = randperm(pool.n); p = 1; end
        ch = pack(p); p = p + 1;
        tries = tries + 1;
        if tries > 20 * pool.n
            error('cannot fill slot %d: pool too small', c);
        end
        if any(here == ch) || any(live == ch), continue; end
        if ~cfg.shareChannels && any(pool.figIdx == ch), continue; end
        here(end+1) = ch; %#ok<AGROW>
    end
    idx = m + (1:starts);
    chan(idx) = here;
    slot(idx) = c;
    isFig(idx(1:nFig)) = true;
    live = [live here]; %#ok<AGROW>
    if numel(live) > k * starts
        live = live(end - k*starts + 1 : end);
    end
    m = m + starts;
end
end

% ------------------------------------------------------------------ render
function out = render(cfg, pool, chan, slot, isFig, k)
% A tone is k hops long plus one hop of ramp, so at every slot boundary the
% tones ramping out are matched by the ones ramping in. With
% power-complementary ramps the total power is constant across the join.
hop = round(cfg.hopMs * cfg.fs / 1000);
n   = k * hop + hop;

t   = (0:n-1) / cfg.fs;
x   = (0:hop-1) / hop;
env = ones(1, n);
env(1:hop)         = sin(pi/2 * x);
env(end-hop+1:end) = cos(pi/2 * x);
pips = sin(2*pi * pool.f(:) * t) .* env;

N  = max(slot) * hop + n;
y  = zeros(1, N);
yF = zeros(1, N);
for i = 1:numel(chan)
    s = (slot(i) - 1) * hop + 1;
    y(s:s+n-1) = y(s:s+n-1) + pips(chan(i), :);
    if isFig(i)
        yF(s:s+n-1) = yF(s:s+n-1) + pips(chan(i), :);
    end
end

% one gain for both, so the figure-only file is exactly the figure you hear
% inside the mix rather than a louder version of it
g = 10^(cfg.peakDbfs/20) / max(abs(y));
out.y      = y * g;
out.figure = yF * g;
out.cloud  = (y - yF) * g;
out.fs     = cfg.fs;
end

% ------------------------------------------------------------------ report
function report(cfg, pool, chan, slot, isFig, out, k, starts)
perSlot = accumarray(slot, 1);
use = accumarray(chan, 1, [pool.n 1]);
isf = false(pool.n, 1); isf(pool.figIdx) = true;
dur = numel(out.y) / cfg.fs;

fprintf('%d Hz | %d-tone figure, %g ms step, %g ms tones at %g Hz\n', ...
    cfg.fs, cfg.nTones, cfg.stepMs, cfg.toneMs, cfg.rateHz);
fprintf('  %g ms grid: step %d slots, jitter +-%d slots, tone %d slots\n', ...
    cfg.hopMs, round(cfg.stepMs/cfg.hopMs), round(cfg.jitterMs/cfg.hopMs), k);
fprintf('  starts per slot %d-%d, so %d tones sounding throughout\n', ...
    min(perSlot), max(perSlot), k * starts);
fprintf('  pool %d channels %.0f-%.0f Hz on a %g st grid\n', ...
    pool.n, pool.f(1), pool.f(end), pool.gridSt);
fprintf('  figure channel %.1f/s vs background %.1f/s: contrast %.1fx\n', ...
    mean(use(isf))/dur, mean(use(~isf))/dur, ...
    mean(use(isf)) / max(mean(use(~isf)), eps));
fprintf('  peak: mix %.1f dBFS, figure alone %.1f dBFS, %.1f s\n', ...
    20*log10(max(abs(out.y))), 20*log10(max(abs(out.figure))), dur);
end

% -------------------------------------------------------------------- plot
function show(cfg, pool, chan, slot, isFig)
nShow = min(max(slot), round(1400 / cfg.hopMs));
sel = slot <= nShow;
t = (slot(sel) - 1) * cfg.hopMs / 1000;
f = reshape(pool.st(chan(sel)), [], 1);
g = isFig(sel);

figure('Color', 'w', 'Position', [100 100 900 520]); hold on
plot(t(~g), f(~g), 's', 'MarkerSize', 4, 'MarkerFaceColor', 'k', ...
    'MarkerEdgeColor', 'none');
plot(t(g), f(g), 's', 'MarkerSize', 4, 'MarkerFaceColor', [0.9 0.1 0.1], ...
    'MarkerEdgeColor', 'none');
xlim([0 nShow * cfg.hopMs / 1000]); ylim(cfg.poolSt + [-2 2]);
xlabel('Time (s)'); ylabel(sprintf('Semitones re %g Hz', cfg.fRefHz));
title(sprintf('%d tones, %g ms step, %g ms tones', ...
    cfg.nTones, cfg.stepMs, cfg.toneMs));
box off
end
