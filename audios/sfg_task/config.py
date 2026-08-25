"""The whole experiment in one object.  Nothing is decided anywhere else."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class Design:
    # --- the manipulation --------------------------------------------
    # How far each successive tone of the figure is delayed behind the one
    # below it.  0 ms is the classic coherent chord; every other level is
    # the same seven tones in the same seven channels, sheared in time.
    steps_ms: tuple[float, ...] = (0.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0)
    coherence: int = 7           # tones in the figure
    order: str = "rise"          # 'rise' | 'fall' -- the main sweep is 'rise'

    # --- time base ----------------------------------------------------
    # Onsets fall on a grid of hop_ms and a tone lasts k = tone_ms/hop_ms
    # slots.  The grid is fixed for the whole experiment, not solved per
    # condition: if it moved with the step, so would the background density,
    # and every step would face a different cloud.  A finer grid costs
    # nothing -- the cloud starts bg_sounding/tone_ms tones a second whatever
    # the grid is -- so it is set by the finest step to be tested.
    fs: int = 48_000
    hop_ms: float = 5.0
    tone_ms: float = 50.0        # as in Teki 2013 / O'Sullivan 2015
    ramp_ms: float = 10.0        # raised cosine, independent of the grid

    # --- the cloud ----------------------------------------------------
    # 179-7246 Hz on a 1/24 octave grid is Teki's pool; 10 tones sounding is
    # their mean chord.  Holding the count fixed is O'Sullivan's control --
    # broadband power and every other low-level feature stay constant.
    f_lo: float = 250.0          # Teki's pool starts at 179 Hz, but they do
    f_hi: float = 7246.0         # not weight for equal loudness; at 60 phon
                                 # that octave is boosted 11 dB and takes
                                 # over the stimulus
    grid_st: float = 0.5
    bg_sounding: int = 8
    dealer_slack: float = 3.0    # counts stay level, order stays unguessable
    guard_ms: float = 50.0       # rest before a channel may sound again, so
                                 # two background tones never abut into one
                                 # long tone and pop out on their own
    min_sep_erb: float = 1.0     # no two tones sounding at once inside one
                                 # critical band.  Without it a 1/24 octave
                                 # pool puts 4-8 channels in every ERB and
                                 # two thirds of the tones acquire a beating
                                 # partner -- at the bottom of the pool that
                                 # is a 5 Hz throb, which is what a listener
                                 # hears as a repeated beep rather than as
                                 # a cloud

    # --- where the figure sits ----------------------------------------
    fig_span_st: float = 30.0    # frequency extent of the figure
    fig_edge_st: float = 8.0     # kept this far clear of both pool edges
    phon: float = 60.0           # ISO 226 weighting, so no region dominates

    # --- one interval -------------------------------------------------
    # Long, because the figure has to be found by accumulating evidence over
    # elements rather than caught in one.  A fixed number of elements per
    # interval, at irregular spacing.
    # Successive elements never overlap -- at 50 ms per step a figure is
    # 350 ms long -- so the number of coherent components sounding at once
    # is the same in every condition.
    interval_s: float = 6.0
    events: int = 11
    lead_ms: float = 300.0
    tail_ms: float = 200.0
    gap_guard_ms: float = 60.0   # silence between the end of one element and
                                 # the onset of the next

    # --- level --------------------------------------------------------
    rms_dbfs: float = -26.0      # calibrate this to 65 dB SPL once
    rove_db: float = 3.0         # +-, drawn per interval, so neither overall
                                 # level nor a level difference is a cue
    edge_ms: float = 50.0        # on/off ramp of the whole interval

    # --- the task -----------------------------------------------------
    task: str = "2ifc"           # '2ifc' (criterion-free) | 'yesno'
    trials_per_step: int = 20    # doubled automatically for yes/no, which
                                 # needs figure-absent trials of its own
    practice_trials: int = 16
    practice_criterion: int = 8  # correct out of the last 10 to move on
    break_every: int = 45
    break_min_s: float = 20.0
    feedback: bool = True
    max_run: int = 3             # never more than this many trials in a row
                                 # at the same step

    # --- trial timing -------------------------------------------------
    ready_ms: float = 400.0
    isi_ms: float = 500.0
    response_s: float = 10.0
    feedback_ms: float = 350.0
    iti_ms: float = 600.0

    # --- controls -----------------------------------------------------
    # Run separately, not mixed into the sweep.
    #   perm    same asynchronies, frozen, but not a monotonic sweep
    #   redraw  same channels and elements, delays redrawn every element
    #   scatter same channels at the same rate, never grouped into elements
    #           -- the long-term spectrum is identical to the figure's, so
    #           this is what separates temporal coherence from a spectral
    #           prominence
    # 'rise' is in the list because the controls are only meaningful next
    # to the sweep measured in the same session, by the same ears.
    control_variants: tuple[str, ...] = ("rise", "perm", "redraw", "scatter")
    control_steps_ms: tuple[float, ...] = (20.0, 40.0)   # 0 ms has no
                                                         # pattern to scramble
    control_trials: int = 20

    seed: int = 20250825

    # ------------------------------------------------------------------
    def replace(self, **kw) -> "Design":
        return replace(self, **kw)

    @property
    def k(self) -> int:
        return int(round(self.tone_ms / self.hop_ms))

    @property
    def density(self) -> float:
        """Background tones starting per slot, on average."""
        return self.bg_sounding / self.k

    @property
    def guard(self) -> int:
        return int(round(self.guard_ms / self.hop_ms))

    @property
    def n_slots(self) -> int:
        return int(round(self.interval_s * 1000.0 / self.hop_ms))

    @property
    def lead(self) -> int:
        return int(round(self.lead_ms / self.hop_ms))

    @property
    def tail(self) -> int:
        return int(round(self.tail_ms / self.hop_ms))

    def extent_ms(self, step_ms: float) -> float:
        """Onset of the first tone to offset of the last."""
        return (self.coherence - 1) * step_ms + self.tone_ms

    @property
    def max_extent(self) -> int:
        """In slots, over every step in the experiment -- so the spacing of
        the elements can be identical in every condition."""
        return int(round(self.extent_ms(max(self.steps_ms)) / self.hop_ms))

    @property
    def min_gap(self) -> int:
        return self.max_extent + int(round(self.gap_guard_ms / self.hop_ms))

    @property
    def slack(self) -> int:
        """Slots left over to spread among the gaps, which is what makes the
        rhythm irregular."""
        first = self.lead
        last = self.n_slots - self.tail - self.max_extent
        return last - first - (self.events - 1) * self.min_gap

    @property
    def n_per_step(self) -> int:
        return self.trials_per_step * (2 if self.task == "yesno" else 1)

    @property
    def n_trials(self) -> int:
        return self.n_per_step * len(self.steps_ms)

    @property
    def trial_s(self) -> float:
        n = 2 if self.task == "2ifc" else 1
        return ((self.ready_ms + self.iti_ms + self.feedback_ms * self.feedback
                 + self.isi_ms * (n - 1)) / 1000.0
                + n * self.interval_s + 1.0)

    def validate(self) -> None:
        if abs(self.tone_ms / self.hop_ms - self.k) > 1e-9:
            raise ValueError("tone_ms must be a whole number of hops")
        for s in self.steps_ms + self.control_steps_ms:
            if s and abs(s / self.hop_ms - round(s / self.hop_ms)) > 1e-9:
                raise ValueError(f"step {s} ms is not a whole number of hops")
        if self.bg_sounding < 1:
            raise ValueError("bg_sounding must be at least 1")
        if self.ramp_ms * 2 > self.tone_ms:
            raise ValueError("ramps do not fit inside the tone")
        if self.slack < 0:
            raise ValueError(
                f"{self.events} elements of {self.extent_ms(max(self.steps_ms)):.0f} ms "
                f"do not fit in {self.interval_s} s: shorten the interval's "
                f"lead/tail, drop an element, or lengthen the interval")
        if self.task not in ("2ifc", "yesno"):
            raise ValueError("task must be '2ifc' or 'yesno'")

    def summary(self) -> str:
        span = 12 * math.log2(self.f_hi / self.f_lo)
        n_ch = int(span / self.grid_st) + 1
        bg = self.density * 1000.0 / self.hop_ms
        rate = self.events / (self.interval_s - (self.lead_ms + self.tail_ms) / 1000)
        lines = [
            f"{self.task}  {len(self.steps_ms)} steps "
            f"{', '.join(f'{s:g}' for s in self.steps_ms)} ms  x "
            f"{self.n_per_step} trials = {self.n_trials} trials",
            f"  pool      {n_ch} channels, {self.f_lo:.0f}-{self.f_hi:.0f} Hz, "
            f"{self.grid_st:g} st grid, {self.phon:g} phon weighting",
            f"  cloud     {self.bg_sounding} tones sounding, "
            f"{bg:.0f} onsets/s, {bg / n_ch:.2f}/s per channel",
            f"  figure    {self.coherence} tones over {self.fig_span_st:g} st, "
            f"{self.events} elements per interval at {rate:.2f}/s "
            f"-> contrast {1 + rate / (bg / n_ch):.1f}x",
            f"  element   {self.extent_ms(min(self.steps_ms)):.0f}-"
            f"{self.extent_ms(max(self.steps_ms)):.0f} ms long, "
            f"gaps {self.min_gap * self.hop_ms:.0f}-"
            f"{(self.min_gap + self.slack) * self.hop_ms:.0f} ms",
            f"  session   {self.n_trials * self.trial_s / 60:.0f} min "
            f"+ practice + {self.n_trials // self.break_every} breaks",
        ]
        return "\n".join(lines)
