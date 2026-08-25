function out = sfg_staircase(varargin)
% Stochastic figure-ground with a staircase figure.
%
% Time is a grid of chords. Every chord holds exactly nChord tones, so the
% envelope is uniform by construction. The figure is a fixed set of channels
% that recurs; stepChords shears it from a chord into a diagonal.
%
%   sfg_staircase                          % 7-tone staircase, one chord/step
%   sfg_staircase('stepChords', 0)         % the classic coherent figure
%   sfg_staircase('stepChords', 2, 'nChord', 2)

cfg = config();
for i = 1:2:numel(varargin)
    cfg.(varargin{i}) = varargin{i+1};
end
rng(cfg.seed);

pool = make_pool(cfg);
[chan, chord, isFig] = schedule(cfg, pool);
out = render(cfg, pool, chan, chord);

report(cfg, pool, chan, chord, isFig, out);
if cfg.doPlot, show(cfg, pool, chan, chord, isFig); end
if ~isempty(cfg.wavFile)
    audiowrite(cfg.wavFile, out.y(:), cfg.fs, 'BitsPerSample', 24);
end
if cfg.doPlay, sound(out.y, cfg.fs); end
end

% ------------------------------------------------------------------ config
function cfg = config()
cfg.fs           = 48000;
cfg.seed         = 3;

cfg.chordMs      = 35;      % the time grid; every tone starts on it
cfg.rampMs       = 5;       % raised-cosine, power-complementary across chords

cfg.nTones       = 7;       % tones in the figure
cfg.stepChords   = 1;       % staircase step, in chords (0 = coherent chord)
cfg.order        = 'rise';  % 'rise' or 'fall'
cfg.spanSt       = 24;      % frequency span of the figure, semitones
cfg.rateHz       = 5;       % figure repetition rate
cfg.jitterChords = 1;       % random displacement of each figure onset
cfg.wobbleChords = 0;       % frozen irregularity of the staircase, in chords

cfg.nChord       = 3;       % tones per chord: the density, and the envelope
cfg.contrast     = 4;       % figure/background per-channel rate, sets the pool
cfg.shareChannels= true;    % let the background use the figure's channels

cfg.fRefHz       = 1000;
cfg.poolSt       = [-24 36];

cfg.durationS    = 20;
cfg.peakDbfs     = -3;
cfg.doPlay       = true;
cfg.doPlot       = true;
cfg.wavFile      = 'sfg_staircase.wav';
end

% -------------------------------------------------------------------- pool
function pool = make_pool(cfg)
% A figure channel sounds rateHz times a second; the background shares what
% is left of the chords between the remaining channels. Asking for a contrast
% therefore fixes how many channels the pool needs.
cloudPerS = cfg.nChord * 1000/cfg.chordMs - cfg.nTones * cfg.rateHz;
want = cloudPerS * cfg.contrast / cfg.rateHz + cfg.nTones;

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
end

% ---------------------------------------------------------------- schedule
function [chan, chord, isFig] = schedule(cfg, pool)
nChords  = floor(cfg.durationS * 1000 / cfg.chordMs);
periodCh = max(1, round(1000 / (cfg.rateHz * cfg.chordMs)));

lag = (0:cfg.nTones-1) * cfg.stepChords;
if strcmp(cfg.order, 'fall'), lag = fliplr(lag); end
if cfg.wobbleChords > 0
    lag = lag + randi([0 cfg.wobbleChords], 1, cfg.nTones);  % frozen
end

% figure first: which of its channels sound in which chord
occ = cell(1, nChords);
for w = 0 : floor(nChords / periodCh)
    c0 = w * periodCh + 1 + randi([-cfg.jitterChords cfg.jitterChords]);
    for k = 1:cfg.nTones
        c = c0 + lag(k);
        if c >= 1 && c <= nChords
            occ{c}(end+1) = pool.figIdx(k);
        end
    end
end

most = max(cellfun(@numel, occ));
if most > cfg.nChord
    error('the figure needs %d tones in one chord but nChord is %d', ...
        most, cfg.nChord);
end

% then fill every chord to exactly nChord, dealing from a shuffled pack so
% the channels stay level, and never repeating one in adjacent chords
chan = zeros(nChords * cfg.nChord, 1);
chord = zeros(size(chan));
isFig = false(size(chan));

pack = randperm(pool.n); p = 1;
prev = [];
m = 0;
for c = 1:nChords
    here = occ{c};
    nFig = numel(here);
    tries = 0;
    while numel(here) < cfg.nChord
        if p > numel(pack), pack = randperm(pool.n); p = 1; end
        k = pack(p); p = p + 1;
        tries = tries + 1;
        if tries > 20 * pool.n
            error('cannot fill chord %d: pool too small', c);
        end
        if any(here == k) || any(prev == k), continue; end
        if ~cfg.shareChannels && any(pool.figIdx == k), continue; end
        here(end+1) = k; %#ok<AGROW>
    end
    idx = m + (1:cfg.nChord);
    chan(idx)  = here;
    chord(idx) = c;
    isFig(idx(1:nFig)) = true;
    prev = here;
    m = m + cfg.nChord;
end
end

% ------------------------------------------------------------------ render
function out = render(cfg, pool, chan, chord)
% Tones are one ramp longer than a chord so consecutive chords cross-fade.
% With power-complementary ramps the total power is then constant across the
% join; abutting tones would leave a dip at every chord boundary.
hop = round(cfg.chordMs * cfg.fs / 1000);
r   = round(cfg.rampMs * cfg.fs / 1000);
n   = hop + r;

t   = (0:n-1) / cfg.fs;
env = ones(1, n);
x   = (0:r-1) / r;
env(1:r)       = sin(pi/2 * x);
env(end-r+1:end) = cos(pi/2 * x);
pips = sin(2*pi * pool.f(:) * t) .* env;

N = (max(chord)) * hop + n;
y = zeros(1, N);
for i = 1:numel(chan)
    s = (chord(i) - 1) * hop + 1;
    y(s:s+n-1) = y(s:s+n-1) + pips(chan(i), :);
end

out.y  = y * (10^(cfg.peakDbfs/20) / max(abs(y)));
out.fs = cfg.fs;
end

% ------------------------------------------------------------------ report
function report(cfg, pool, chan, chord, isFig, out)
perChord = accumarray(chord, 1);
use = accumarray(chan, 1, [pool.n 1]);
isf = false(pool.n, 1); isf(pool.figIdx) = true;
dur = numel(out.y) / cfg.fs;

fprintf('%d Hz | %d-tone figure, step %d chord(s), %g ms chords at %g Hz\n', ...
    cfg.fs, cfg.nTones, cfg.stepChords, cfg.chordMs, cfg.rateHz);
fprintf('  pool %d channels %.0f-%.0f Hz on a %g st grid\n', ...
    pool.n, pool.f(1), pool.f(end), pool.gridSt);
fprintf('  tones per chord %d-%d (uniform envelope requires one value)\n', ...
    min(perChord), max(perChord));
fprintf('  figure channel %.1f/s vs background %.1f/s: contrast %.1fx\n', ...
    mean(use(isf))/dur, mean(use(~isf))/dur, ...
    mean(use(isf)) / max(mean(use(~isf)), eps));
fprintf('  channel use %d-%d, %d figure tones of %d, %.1f s, peak %.1f dBFS\n', ...
    min(use), max(use), sum(isFig), numel(chan), dur, ...
    20*log10(max(abs(out.y))));
end

% -------------------------------------------------------------------- plot
function show(cfg, pool, chan, chord, isFig)
nShow = min(max(chord), round(1400 / cfg.chordMs));
k = chord <= nShow;
t = (chord(k) - 1) * cfg.chordMs / 1000;
f = pool.st(chan(k))';
g = isFig(k);

figure('Color', 'w', 'Position', [100 100 900 520]); hold on
plot(t(~g), f(~g), 's', 'MarkerSize', 4, 'MarkerFaceColor', 'k', ...
    'MarkerEdgeColor', 'none');
plot(t(g), f(g), 's', 'MarkerSize', 4, 'MarkerFaceColor', [0.9 0.1 0.1], ...
    'MarkerEdgeColor', 'none');
xlim([0 nShow * cfg.chordMs / 1000]); ylim(cfg.poolSt + [-2 2]);
xlabel('Time (s)'); ylabel(sprintf('Semitones re %g Hz', cfg.fRefHz));
title(sprintf('%d tones, step %d chord(s), %d per chord', ...
    cfg.nTones, cfg.stepChords, cfg.nChord));
box off
end
