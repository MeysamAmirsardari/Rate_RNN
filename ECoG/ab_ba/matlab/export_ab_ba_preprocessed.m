function export_ab_ba_preprocessed(source_root, output_file, experiments)
%EXPORT_AB_BA_PREPROCESSED Lossless bridge from ft_oe_list to Python.
%
% This function intentionally performs only the source preprocessing and
% source row selection. Decoder fitting happens in Python, where the original
% and leakage-safe cross-validation profiles are both auditable.
%
% Example:
%   export_ab_ba_preprocessed( ...
%       '/Users/eminent/Projects/ECoG/AB_BA', ...
%       '/Users/eminent/Projects/ECoG/AB_BA/ab_ba_preprocessed_export.mat');
%
% Export experiment 1 only (the 180-ms, zero-gap condition):
%   export_ab_ba_preprocessed( ...
%       '/Users/eminent/Projects/ECoG/AB_BA', ...
%       '/Users/eminent/Projects/ECoG/AB_BA/ab_ba_preprocessed_export.mat', ...
%       1);
%
% Requirements on the MATLAB path:
%   ft_oe_list.m, Gen_M2Mat.m, and the Open Ephys helper dependencies used in
%   the original analysis environment.

if nargin < 1 || isempty(source_root)
    projects_root = fileparts(fileparts(fileparts(fileparts(fileparts( ...
        mfilename('fullpath'))))));
    source_root = fullfile(projects_root, 'ECoG', 'AB_BA');
end
if nargin < 2 || isempty(output_file)
    output_file = fullfile(source_root, 'ab_ba_preprocessed_export.mat');
end
if nargin < 3 || isempty(experiments)
    experiments = 1:3;
end
validateattributes(experiments, {'numeric'}, ...
    {'vector', 'integer', '>=', 1, '<=', 3, 'nonempty'}, ...
    mfilename, 'experiments', 3);
experiments = unique(experiments(:)', 'stable');

required = {'ft_oe_list', 'Gen_M2Mat'};
for ii = 1:numel(required)
    assert(exist(required{ii}, 'file') == 2, ...
        'ABBA:MissingDependency', 'Required MATLAB function is absent: %s.m', required{ii});
end

old_dir = pwd;
cleanup = onCleanup(@() cd(old_dir));
cd(source_root);

runclass = 'SEQ';
xx1 = ft_oe_list('Nutmeg_2026-04-30', runclass, [1 250]);
xx2 = ft_oe_list('Nutmeg_2026-05-01', runclass, [1 250]);
allMatrices1 = build_all_matrices(xx1, experiments);
allMatrices2 = build_all_matrices(xx2, experiments);

comparisons = struct();
for expnum = experiments
    for deviant_day = 1:2
        key = sprintf('exp%d_day%d_deviant', expnum, deviant_day);
        comparisons.(key) = make_comparison( ...
            allMatrices1, allMatrices2, expnum, deviant_day);
    end
end

export_metadata = struct();
export_metadata.created = char(datetime('now', 'TimeZone', 'local', ...
    'Format', 'yyyy-MM-dd''T''HH:mm:ssXXX'));
export_metadata.source_root = source_root;
export_metadata.output_file = output_file;
export_metadata.source_dates = {'Nutmeg_2026-04-30', 'Nutmeg_2026-05-01'};
export_metadata.runclass = runclass;
export_metadata.filter_argument_hz = [1 250];
export_metadata.loader = 'Gen_M2Mat';
export_metadata.hilbert_t = 0;
export_metadata.outlier_removal = 0;
export_metadata.class_0 = 'deviant sequence';
export_metadata.class_1 = ['same physical sequence presented as a standard ' ...
    'immediately after the opposite deviant on the other day'];
export_metadata.row_selection = ['allM2(allM2(:,4)==rare_stimulus,:); ' ...
    'allM2(find(allM2(:,4)==other_day_rare_stimulus)+1,:)'];
export_metadata.selected_experiments = experiments;
if isequal(experiments, 1:3)
    export_metadata.export_scope = 'all experiments';
else
    export_metadata.export_scope = sprintf('experiments %s', mat2str(experiments));
end

save(output_file, 'comparisons', 'export_metadata', '-v7');
fprintf('Saved AB/BA preprocessing export: %s\n', output_file);
end


function allMatrices = build_all_matrices(xx, experiments)
allMatrices = cell(3, 32);
for expnum = experiments
    for ch = 1:32
        item = Gen_M2Mat(xx, expnum, ch, 0);
        if expnum == 1 || expnum == 2
            item.info.noteDur = 180;
            item.info.noteGap = 0;
        else
            item.info.noteDur = 50;
            item.info.noteGap = 100;
        end
        allMatrices{expnum, ch} = item;
    end
end
end


function output = make_comparison(day1, day2, expnum, deviant_day)
days = {day1, day2};
standard_day = 3 - deviant_day;
dev_day_matrices = days{deviant_day};
std_day_matrices = days{standard_day};
dev_item = dev_day_matrices{expnum, 1};
std_item = std_day_matrices{expnum, 1};

[dev_stim, dev_counts] = rare_stimulus(dev_item);
[other_deviant_stim, std_counts] = rare_stimulus(std_item);
dev_rows = find(dev_item.allM2(:, 4) == dev_stim);
std_rows = find(std_item.allM2(:, 4) == other_deviant_stim) + 1;
assert(all(std_rows <= size(std_item.allM2, 1)), ...
    'ABBA:SourceRowOverflow', ...
    'The literal find(...)+1 selection exceeds allM2 for exp %d day %d.', ...
    expnum, standard_day);
std_predecessor_rows = std_rows - 1;
assert(all(std_item.allM2(std_predecessor_rows, 6) == ...
           std_item.allM2(std_rows, 6)), ...
    'ABBA:CrossTrialSuccessor', ...
    ['The literal find(rare)+1 selection crosses an acquisition-trial ' ...
     'boundary. Stopping rather than pairing observations across blocks.']);

dev_names = stimulus_names(dev_item.info.stimList);
std_names = stimulus_names(std_item.info.stimList);
target_sequence = dev_names{dev_stim};
target_std_stim = find(strcmpi(std_names, target_sequence), 1);
assert(~isempty(target_std_stim), 'ABBA:MissingPhysicalMatch', ...
    'Target %s is absent from the opposite-day stimulus list.', target_sequence);
assert(all(std_item.allM2(std_rows, 4) == target_std_stim), ...
    'ABBA:InvalidStandardRows', ...
    ['The source find(rare)+1 rows are not all the same physical sequence. ' ...
     'Stopping rather than silently changing the MATLAB method.']);

n_deviant = numel(dev_rows);
n_standard = numel(std_rows);
n_keep = min(n_deviant, n_standard);
dev_rows = dev_rows(1:n_keep);
std_rows = std_rows(1:n_keep);

baseline = dev_item.info.basetime;
win_length = dev_item.info.seqDur + 1000;
assert(baseline == std_item.info.basetime, 'ABBA:BaselineMismatch');
assert(win_length == std_item.info.seqDur + 1000, 'ABBA:WindowMismatch');
erp_win = baseline + (1:win_length);

x_deviant = zeros(32, n_keep, win_length);
x_standard_after_deviant = zeros(32, n_keep, win_length);
for ch = 1:32
    dev_m = dev_day_matrices{expnum, ch}.allM2;
    std_m = std_day_matrices{expnum, ch}.allM2;
    assert(isequal(dev_m(:, 1:6), dev_item.allM2(:, 1:6)), ...
        'ABBA:ChannelMetadataMismatch');
    assert(isequal(std_m(:, 1:6), std_item.allM2(:, 1:6)), ...
        'ABBA:ChannelMetadataMismatch');
    assert(6 + erp_win(end) <= size(dev_m, 2), 'ABBA:ShortDeviantEpoch');
    assert(6 + erp_win(end) <= size(std_m, 2), 'ABBA:ShortStandardEpoch');
    x_deviant(ch, :, :) = dev_m(dev_rows, 6 + erp_win);
    x_standard_after_deviant(ch, :, :) = std_m(std_rows, 6 + erp_win);
end

output = struct();
output.x_deviant = x_deviant;
output.x_standard_after_deviant = x_standard_after_deviant;
output.deviant_trials = dev_item.allM2(dev_rows, 6)';
output.standard_trials = std_item.allM2(std_rows, 6)';
output.deviant_groups = (deviant_day * 100000 + output.deviant_trials);
output.standard_groups = (standard_day * 100000 + output.standard_trials);
output.deviant_source_rows_matlab = dev_rows';
output.standard_source_rows_matlab = std_rows';
output.time_ms = 0:(win_length - 1);
output.source_time_labels_ms = 1:win_length;
output.target_sequence = target_sequence;
output.expnum = expnum;
output.deviant_day = deviant_day;
output.standard_source_day = standard_day;
output.deviant_stimulus_index_matlab = dev_stim;
output.standard_stimulus_index_matlab = target_std_stim;
output.other_day_deviant_stimulus_index_matlab = other_deviant_stim;
output.n_deviant_before_balance = n_deviant;
output.n_standard_before_balance = n_standard;
output.n_keep_per_class = n_keep;
output.deviant_stimulus_counts = dev_counts;
output.standard_day_stimulus_counts = std_counts;
output.baseline_samples = baseline;
output.sequence_duration_samples = dev_item.info.seqDur;
output.note_duration_ms = dev_item.info.noteDur;
output.note_gap_ms = dev_item.info.noteGap;
end


function [index, counts] = rare_stimulus(item)
n_stim = numel(stimulus_names(item.info.stimList));
counts = accumarray(item.allM2(:, 4), 1, [n_stim, 1]);
[~, index] = min(counts);
if isfield(item.info, 'pctList')
    [~, pct_index] = min(item.info.pctList);
    assert(index == pct_index, 'ABBA:RareStimulusMismatch', ...
        'pctList and allM2 counts identify different rare stimuli.');
end
end


function names = stimulus_names(value)
if isstruct(value) && isfield(value, 'stims')
    value = value.stims;
end
if isstring(value)
    names = cellstr(value);
elseif iscell(value)
    names = cellfun(@(x) char(string(x)), value, 'UniformOutput', false);
else
    names = cellstr(string(value));
end
names = names(:)';
end
