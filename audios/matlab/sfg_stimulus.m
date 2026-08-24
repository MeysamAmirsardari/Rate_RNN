function out = sfg_stimulus(varargin)
% Figure-ground stimulus: a frozen tone pattern repeated inside a tone cloud.
% Generates, plays, plots and saves. Pass name/value pairs to override config.
%
%   sfg_stimulus                                  % coherent figure
%   sfg_stimulus('toneStepMs', 10)                % sheared into a staircase
%   sfg_stimulus('flatEnvelope', true)            % hold the total constant

cfg = config();
for i = 1:2:numel(varargin)
    cfg.(varargin{i}) = varargin{i+1};
end
rng(cfg.seed);

check_flat(cfg);
pool = make_pool(cfg);
fig  = make_figure(cfg, pool);
cld  = make_cloud(cfg, pool, fig);
out  = render(cfg, pool, fig, cld);

report(cfg, pool, fig, cld, out);
if cfg.doPlot, show(cfg, pool, fig, cld, out); end
if ~isempty(cfg.wavFile)
    audiowrite(cfg.wavFile, out.mix(:), cfg.fs, 'BitsPerSample', 24);
end
if cfg.doPlay, sound(out.mix, cfg.fs); end
end

% ------------------------------------------------------------------ config
function cfg = config()
cfg.fs             = 48000;
cfg.seed           = 3;

% --- tone
cfg.toneMs         = 25;      % duration of every tone, figure and cloud
cfg.rampMs         = 5;       % raised-cosine onset/offset

% --- figure
cfg.nTones         = 10;      % tones per syllable
cfg.nSyllables     = 1;       % syllables per word
cfg.rateHz         = 5;       % word repetition rate
cfg.toneStepMs     = 0;       % shear between successive tones (0 = chord)
cfg.syllableStepMs = 200;     % onset spacing between syllables
cfg.order          = 'rise';  % 'rise' or 'fall' staircase
cfg.spanSt         = 24;      % frequency span of one syllable, semitones
cfg.syllableStepSt = 3;       % transposition between syllables

% --- jitter
cfg.toneJitterMs   = 0;       % random offset of each tone within a syllable
cfg.wordJitterMs   = 40;      % random displacement of each whole word
cfg.freezeInternal = true;    % draw the within-word jitter once and reuse it

% --- cloud
cfg.cloudTones     = 3;       % tones sounding at once: the sparsity control
cfg.flatEnvelope   = true;    % true: cloud fills the figure's gaps so the
                              % total never moves. See check_flat below for
                              % the two conditions this needs.
cfg.shareChannels  = false;   % let the cloud use the figure's own channels
cfg.contrast       = 4;       % figure/background per-channel rate ratio,
                              % which sets the pool size
cfg.dealerSlack    = 5;       % looseness of channel balancing; 1 makes the
                              % background regular, which competes with the
                              % figure
cfg.guardMs        = 25;      % silence forced around a channel's own tones

% --- pool
cfg.fRefHz         = 1000;
cfg.poolSt         = [-24 36];

% --- presentation
cfg.durationS      = 20;
cfg.leadMs         = 400;
cfg.tailMs         = 600;
cfg.peakDbfs       = -3;
cfg.doPlay         = true;
cfg.doPlot         = true;
cfg.wavFile        = 'sfg_stimulus.wav';
end

% ------------------------------------------------------------------- check
function check_flat(cfg)
% A uniform envelope needs the cloud to fill whatever the figure leaves, and
% it can only do that with tones of its own length. Two things follow.
if ~cfg.flatEnvelope, return; end

% The figure's own profile must be piecewise constant on tone-length blocks,
% which means the shear is 0 or a whole number of tones. Anything between
% makes the figure's concurrency change faster than the cloud can track:
% measured at a 25 ms tone, a 10 ms shear leaves the total below target 26%
% of the time and a 20 ms shear 49%, against under 1% at 0, 25 or 50 ms.
r = mod(cfg.toneStepMs, cfg.toneMs);
if r > 1e-9
    warning('sfg:shear', ...
        ['toneStepMs %g is not a multiple of toneMs %g, so the envelope ' ...
         'cannot be uniform. Use %g or %g.'], ...
        cfg.toneStepMs, cfg.toneMs, ...
        floor(cfg.toneStepMs/cfg.toneMs)*cfg.toneMs, ...
        ceil(cfg.toneStepMs/cfg.toneMs)*cfg.toneMs);
end

% The total can never be below the figure's own peak, so a coherent n-tone
% chord forces an n-tone background however sparse you asked for.
if cfg.toneStepMs < cfg.toneMs
    peak = cfg.nTones;
else
    peak = max(1, ceil(cfg.toneMs / max(cfg.toneStepMs, eps)));
end
if peak > cfg.cloudTones
    warning('sfg:density', ...
        ['the figure peaks at %d tones, so a uniform envelope needs %d ' ...
         'in the background, not the %d requested.'], ...
        peak, peak, cfg.cloudTones);
end
end

% -------------------------------------------------------------------- pool
function pool = make_pool(cfg)
% Channel count follows from the contrast you ask for: a figure channel
% sounds rateHz times a second, a background one gets the cloud spread over
% the remaining channels.
nFig  = cfg.nTones * cfg.nSyllables;
toneS = cfg.toneMs / 1000;
want  = cfg.cloudTones * cfg.contrast / (toneS * cfg.rateHz) + nFig;

grids  = 0.25:0.25:12;
counts = floor(diff(cfg.poolSt) ./ grids) + 1;
[~, i] = min(abs(counts - want));

pool.gridSt = grids(i);
pool.st     = cfg.poolSt(1):pool.gridSt:cfg.poolSt(2);
pool.f      = cfg.fRefHz * 2.^(pool.st / 12);
pool.n      = numel(pool.st);

% figure channels: evenly spread across the middle, so they always land on
% the grid however many there are
lo   = round(0.15 * (pool.n - 1)) + 1;
span = round(cfg.spanSt / pool.gridSt);
step = round(cfg.syllableStepSt / pool.gridSt);
if lo + span + (cfg.nSyllables-1)*step > pool.n
    error('figure spans %g st beyond the %g st pool', ...
        cfg.spanSt + (cfg.nSyllables-1)*cfg.syllableStepSt, diff(cfg.poolSt));
end
base = round(linspace(lo, lo + span, cfg.nTones))';
pool.figIdx = base + (0:cfg.nSyllables-1) * step;   % nTones x nSyllables
end

% ------------------------------------------------------------------ figure
function fig = make_figure(cfg, pool)
period = 1000 / cfg.rateHz;
nWords = max(1, round(cfg.durationS * cfg.rateHz));
lead   = max(cfg.leadMs, cfg.wordJitterMs + cfg.toneJitterMs + cfg.toneMs);

lag = (0:cfg.nTones-1)' * cfg.toneStepMs;
if strcmp(cfg.order, 'fall'), lag = flipud(lag); end
lag = lag + (0:cfg.nSyllables-1) * cfg.syllableStepMs;   % nTones x nSyl

frozen = (2*rand(size(lag)) - 1) * cfg.toneJitterMs;

% A word onset quantised to whole tone lengths keeps the cloud's own tiling
% aligned with it, which is what lets the flat mode stay flat.
if cfg.flatEnvelope && cfg.wordJitterMs >= cfg.toneMs
    k = floor(cfg.wordJitterMs / cfg.toneMs);
    jitter = randi([-k k], 1, nWords) * cfg.toneMs;
else
    jitter = (2*rand(1, nWords) - 1) * cfg.wordJitterMs;
end

chan = []; onset = []; word = [];
for w = 1:nWords
    t0 = lead + (w-1) * period + jitter(w);
    if cfg.freezeInternal
        dj = frozen;
    else
        dj = (2*rand(size(lag)) - 1) * cfg.toneJitterMs;
    end
    t = round(t0 + lag + dj);
    chan  = [chan;  pool.figIdx(:)];
    onset = [onset; t(:)];
    word  = [word;  repmat(w, numel(t), 1)];
end

fig.chan    = chan;
fig.onsetMs = onset;
fig.word    = word;
fig.nWords  = nWords;
fig.period  = period;
fig.totalMs = max(onset) + cfg.toneMs + cfg.tailMs;
end

% ------------------------------------------------------------------- cloud
function cld = make_cloud(cfg, pool, fig)
T     = fig.totalMs;
tone  = cfg.toneMs;
guard = cfg.guardMs;

conc = zeros(1, T);
busy = false(pool.n, T);

for i = 1:numel(fig.onsetMs)
    o = fig.onsetMs(i);
    if cfg.flatEnvelope
        conc(o : o+tone-1) = conc(o : o+tone-1) + 1;
    end
    g = max(1, o-guard) : min(T, o+tone-1+guard);
    busy(fig.chan(i), g) = true;
end

if cfg.flatEnvelope
    target = max(max(conc), cfg.cloudTones);
else
    target = cfg.cloudTones;
end

% The cloud is kept off the figure's channels by default. Sharing them means
% a cloud tone can only fall in the gap between two figure tones, which is
% heard as a beat halfway through the figure's period.
if ~cfg.shareChannels
    busy(unique(fig.chan), :) = true;
end

seen = zeros(1, pool.n);
chan = zeros(0, 1); onset = zeros(0, 1);

for o = 1 : T - tone + 1
    w = o : o+tone-1;
    while max(conc(w)) < target
        [~, ord] = sort(floor(seen / cfg.dealerSlack) + rand(1, pool.n));
        k = 0;
        for c = ord
            if ~any(busy(c, w)), k = c; break; end
        end
        if k == 0, break; end
        conc(w) = conc(w) + 1;
        busy(k, max(1, o-guard) : min(T, o+tone-1+guard)) = true;
        seen(k) = seen(k) + 1;
        chan(end+1, 1)  = k;      %#ok<AGROW>
        onset(end+1, 1) = o;      %#ok<AGROW>
    end
end

cld.chan    = chan;
cld.onsetMs = onset;
end

% ------------------------------------------------------------------ render
function out = render(cfg, pool, fig, cld)
n = round(cfg.toneMs * cfg.fs / 1000);
N = round(fig.totalMs * cfg.fs / 1000) + n;

t    = (0:n-1) / cfg.fs;
pips = sin(2*pi * pool.f(:) * t) .* gate(n, cfg.rampMs, cfg.fs);

yFig = add_tones(zeros(1, N), pips, fig.chan, fig.onsetMs, cfg.fs, n);
yCld = add_tones(zeros(1, N), pips, cld.chan, cld.onsetMs, cfg.fs, n);

% one gain for both, so the figure-only file is exactly the figure you hear
% inside the mix
g = 10^(cfg.peakDbfs/20) / max(abs(yFig + yCld));

out.mix    = (yFig + yCld) * g;
out.figure = yFig * g;
out.cloud  = yCld * g;
out.fs     = cfg.fs;
end

function y = add_tones(y, pips, chan, onsetMs, fs, n)
for i = 1:numel(chan)
    s = round(onsetMs(i) * fs / 1000) + 1;
    y(s : s+n-1) = y(s : s+n-1) + pips(chan(i), :);
end
end

function env = gate(n, rampMs, fs)
r = round(rampMs * fs / 1000);
env = ones(1, n);
w = sin(pi * (0:r-1) / (2*r)).^2;
env(1:r) = w;
env(end-r+1:end) = fliplr(w);
end

% ------------------------------------------------------------------ report
function report(cfg, pool, fig, cld, out)
tone = cfg.toneMs;
T    = fig.totalMs;

c = zeros(1, T);
for o = [fig.onsetMs; cld.onsetMs]', c(o:o+tone-1) = c(o:o+tone-1) + 1; end
inner = c(cfg.leadMs : T - cfg.tailMs);

use = accumarray([fig.chan; cld.chan], 1, [pool.n 1]);
isFig = false(pool.n, 1); isFig(unique(fig.chan)) = true;
dur = fig.nWords * fig.period / 1000;

fprintf('%d Hz | %d tones x %d syllables at %g Hz, %g ms tones\n', ...
    cfg.fs, cfg.nTones, cfg.nSyllables, cfg.rateHz, tone);
fprintf('  shear %g ms (%s), word jitter +-%g ms, tone jitter +-%g ms\n', ...
    cfg.toneStepMs, cfg.order, cfg.wordJitterMs, cfg.toneJitterMs);
fprintf('  pool %d channels %.0f-%.0f Hz on a %g st grid\n', ...
    pool.n, pool.f(1), pool.f(end), pool.gridSt);
fprintf('  concurrency %d-%d, mean %.2f +- %.2f%s\n', ...
    min(inner), max(inner), mean(inner), std(inner), ...
    ternary(cfg.flatEnvelope, ' (flat)', ''));
fprintf('  figure channel %.1f/s vs background %.1f/s: contrast %.1fx\n', ...
    mean(use(isFig))/dur, mean(use(~isFig))/dur, ...
    mean(use(isFig)) / max(mean(use(~isFig)), eps));
fprintf('  peak %.2f dBFS, %.1f s\n', ...
    20*log10(max(abs(out.mix))), numel(out.mix)/cfg.fs);
end

function s = ternary(c, a, b)
if c, s = a; else, s = b; end
end

% -------------------------------------------------------------------- plot
function show(cfg, pool, fig, cld, out)
win = min(fig.totalMs, cfg.leadMs + 4000);
tone = cfg.toneMs;

figure('Color', 'w', 'Position', [100 100 1100 620]);

subplot(3, 1, [1 2]); hold on
draw(cld.chan, cld.onsetMs, pool.st, tone, win, [0 0 0], 2.5);
draw(fig.chan, fig.onsetMs, pool.st, tone, win, [0.91 0.07 0.10], 3.5);
xlim([0 win/1000]); ylim(cfg.poolSt + [-2 2]);
ylabel(sprintf('Semitones re %g Hz', cfg.fRefHz));
title(sprintf('shear %g ms, cloud %d tones, contrast %gx', ...
    cfg.toneStepMs, cfg.cloudTones, cfg.contrast));
box off

subplot(3, 1, 3);
c = zeros(1, fig.totalMs);
for o = [fig.onsetMs; cld.onsetMs]', c(o:o+tone-1) = c(o:o+tone-1) + 1; end
plot((1:win)/1000, c(1:win), 'k', 'LineWidth', 1);
xlim([0 win/1000]); ylim([0 max(c)+1]);
xlabel('Time (s)'); ylabel('Tones sounding'); box off
end

function draw(chan, onsetMs, st, tone, win, col, lw)
k = onsetMs < win;
x = [onsetMs(k)'; onsetMs(k)' + tone; nan(1, sum(k))] / 1000;
y = repmat(reshape(st(chan(k)), 1, []), 3, 1); y(3, :) = nan;
plot(x(:), y(:), 'Color', col, 'LineWidth', lw);
end
