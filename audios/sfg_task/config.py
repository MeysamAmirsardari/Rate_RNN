"""The whole experiment in one object.  Nothing is decided anywhere else."""

from __future__ import annotations

import math

import numpy as np
from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class Design:
    # --- the manipulation --------------------------------------------
    # How far each successive tone of the figure is delayed behind the one
    # below it.  0 ms is the classic coherent chord; every other level is
    # the same seven tones in the same seven channels, sheared in time.
    # 0 ms is not in the sweep.  A simultaneous chord would need `coherence`
    # tones starting in one slot, and since the figure substitutes background
    # tones rather than adding to them, that means 1400 onsets a second and
    # no contrast left.  5 ms is the anchor instead: 30 ms from the figure's
    # first tone to its last, well inside any integration window.
    steps_ms: tuple[float, ...] = (5.0, 15.0, 25.0, 35.0, 45.0)
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
    tone_ms: float = 30.0        # 25-50 ms in the literature
    ramp_ms: float = 10.0        # raised cosine, independent of the grid

    # --- the cloud ----------------------------------------------------
    # 179-7246 Hz on a 1/24 octave grid is Teki's pool; 10 tones sounding is
    # their mean chord.  Holding the count fixed is O'Sullivan's control --
    # broadband power and every other low-level feature stay constant.
    f_lo: float = 250.0          # Teki's pool starts at 179 Hz, but they do
    f_hi: float = 7246.0         # not weight for equal loudness; at 60 phon
                                 # that octave is boosted 11 dB and takes
                                 # over the stimulus
    grid_st: float = 0.25        # 1/48 octave, finer than the published 1/24.
                                 # The background's rate per channel is what
                                 # the figure has to stand out from, and a
                                 # finer grid thins it without touching the
                                 # figure: contrast 2.7x at 0.5 st, 4.5x at
                                 # 0.25.  Nothing beats, because no two tones
                                 # sound at once inside one critical band
                                 # whatever the grid is.
    bg_sounding: int = 6         # TOTAL tones sounding, figure included: the
                                 # figure substitutes, so this is the whole
                                 # cloud.  It has to be tone_ms/hop_ms, which
                                 # is one start per slot: fewer cannot cover
                                 # every slot, more raises the onset rate and
                                 # costs contrast for nothing.
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

    # ISO 226 corrects for the threshold of hearing, and nothing here is
    # anywhere near threshold: measured, the masking the cloud throws at
    # every frequency in this pool sits 30 to 45 dB above the threshold, so
    # masking and not audibility is what limits every channel.  The
    # correction therefore buys no evenness at all -- audibility spreads
    # 6.0 dB across the pool without it and 6.9 dB with it -- while costing
    # 12.6 dB of level range, putting the two loudest bands at the two pool
    # edges where least masks them (which is what a listener hears as a
    # beep), and leaving the seven tones of an element up to 12.6 dB apart
    # when they are meant to bind into one thing.  Flat, as in Teki 2013.
    equal_loudness: bool = False
    phon: float = 60.0           # only when equal_loudness is on

    # --- one interval -------------------------------------------------
    # Long, because the figure has to be found by accumulating evidence over
    # elements rather than caught in one.
    #
    # Elements repeat at `rate_hz` and ARE ALLOWED TO OVERLAP.  At 5 Hz the
    # period is 200 ms and a 50 ms-per-step element is 350 ms long, so at
    # the wide delays nearly two elements sound at once and the figure is a
    # continuous ascending stream rather than a discrete object.  The number
    # of coherent components sounding therefore varies with delay, which the
    # non-overlapping version was built to avoid.  It is matched between the
    # two intervals of a trial, so it cannot be a cue, but it does mean the
    # conditions differ in more than asynchrony.  Chosen deliberately, to
    # keep the figure at a speech rate.
    interval_s: float = 6.0
    rate_hz: float = 3.0         # elements NEVER overlap: at 5 Hz a 300 ms
                                 # element ran into the next one and tones
                                 # from successive repetitions landed on top
                                 # of each other, which put accidental
                                 # synchrony back into exactly the delays
                                 # meant to have none
    jitter_ms: float = 20.0      # +- on each interval between elements, so
                                 # the rhythm is not isochronous
    element_gap_ms: float = 20.0  # silence between one element and the next
    lead_ms: float = 300.0
    tail_ms: float = 200.0

    # --- level --------------------------------------------------------
    rms_dbfs: float = -26.0      # calibrate this to 65 dB SPL once
    rove_db: float = 3.0         # +-, drawn per interval, so neither overall
                                 # level nor a level difference is a cue
    edge_ms: float = 50.0        # on/off ramp of the whole interval

    # --- the task -----------------------------------------------------
    # What the listener is comparing the figure against.  'scattered' is the
    # same seven channels ungrouped, which is the only version in which an
    # observer who ignores time cannot do the task; 'cloud' is the classic
    # figure-absent interval, which such an observer solves at 100%.
    absent: str = "scattered"    # 'scattered' | 'cloud'

    task: str = "2ifc"           # '2ifc' (criterion-free) | 'yesno'
    trials_per_step: int = 24    # doubled automatically for yes/no, which
                                 # needs figure-absent trials of its own
    practice_trials: int = 16
    practice_criterion: int = 8  # correct out of the last 10 to move on
    break_every: int = 45
    break_min_s: float = 20.0
    feedback: bool = True
    show_step: bool = False      # printing the condition on the trial line
                                 # tells the subject how hard the trial is
                                 # before they hear it, and they will use it.
                                 # On only when you are testing the runner
                                 # yourself.
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
        """In slots, over every step in the experiment -- so the room left
        for the elements is identical in every condition."""
        return int(round(self.extent_ms(max(self.steps_ms)) / self.hop_ms))

    @property
    def span(self) -> int:
        """Slots between the first element onset and the last."""
        return self.n_slots - self.tail - self.max_extent - self.lead

    @property
    def min_gap(self) -> int:
        """The shortest allowed interval between two element onsets.

        Long enough that the longest element in the experiment finishes
        first.  Elements that overlap put tones from successive repetitions
        on top of each other, and because the delays divide into the period
        that restores synchrony at the widest delays: the manipulation stops
        being monotonic.  Also at least one tone plus its rest, since every
        element uses the same channels."""
        return max(self.max_extent + int(round(self.element_gap_ms / self.hop_ms)),
                   int(round((self.tone_ms + self.guard_ms) / self.hop_ms)))

    @property
    def events(self) -> int:
        """Elements per interval: the rate, or as many as leave room for the
        jitter, whichever is fewer."""
        by_rate = int(round(self.rate_hz * self.span * self.hop_ms / 1000)) + 1
        room = self.min_gap + int(round(self.jitter_ms / self.hop_ms))
        return max(2, min(by_rate, self.span // room + 1))

    @property
    def realised_rate(self) -> float:
        return (self.events - 1) / (self.span * self.hop_ms / 1000)

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
        if self.span < (self.events - 1) * self.min_gap:
            raise ValueError(
                f"{self.events} elements at {self.rate_hz:g} Hz need gaps of "
                f"{self.min_gap * self.hop_ms:.0f} ms and there is only "
                f"{self.span * self.hop_ms:.0f} ms of room: slow the rate, "
                f"shorten the tones, or lengthen the interval")
        # The figure substitutes background tones, so a slot can only hold
        # as many figure tones as it holds tones.  A delay smaller than one
        # slot puts the whole element in one slot.
        # Above one slot the element's tones land in different slots, and
        # the rare collision between two overlapping elements is absorbed as
        # a fraction of a decibel.
        need = self.coherence if min(self.steps_ms) < self.hop_ms else 1
        if self.density < need:
            raise ValueError(
                f"a {min(self.steps_ms):g} ms delay puts {need} figure tones "
                f"in one slot and only {self.density:g} tones start per slot. "
                f"Raise bg_sounding to {int(np.ceil(need * self.k))} (which "
                f"costs contrast), raise hop_ms, or drop that delay from "
                f"steps_ms")
        if self.absent not in ("scattered", "cloud"):
            raise ValueError("absent must be 'scattered' or 'cloud'")
        if self.task not in ("2ifc", "yesno"):
            raise ValueError("task must be '2ifc' or 'yesno'")

    def summary(self) -> str:
        span = 12 * math.log2(self.f_hi / self.f_lo)
        n_ch = int(span / self.grid_st) + 1
        bg = self.density * 1000.0 / self.hop_ms
        rate = self.realised_rate
        lines = [
            f"{self.task}  {len(self.steps_ms)} steps "
            f"{', '.join(f'{s:g}' for s in self.steps_ms)} ms  x "
            f"{self.n_per_step} trials = {self.n_trials} trials",
            f"  pool      {n_ch} channels, {self.f_lo:.0f}-{self.f_hi:.0f} Hz, "
            f"{self.grid_st:g} st grid, "
            + (f"{self.phon:g} phon weighted" if self.equal_loudness
               else "flat level"),
            f"  cloud     {self.bg_sounding} tones sounding, "
            f"{bg:.0f} onsets/s, {bg / n_ch:.2f}/s per channel",
            f"  figure    {self.coherence} tones over {self.fig_span_st:g} st, "
            f"{self.events} elements per interval at {rate:.2f} Hz "
            f"+-{self.jitter_ms:g} ms -> contrast "
            f"{1 + self.events / self.interval_s / (bg / n_ch):.1f}x",
            f"  element   {self.extent_ms(min(self.steps_ms)):.0f}-"
            f"{self.extent_ms(max(self.steps_ms)):.0f} ms long, gaps never "
            f"under {self.min_gap * self.hop_ms:.0f} ms, so they never "
            f"overlap",
            f"  session   {self.n_trials * self.trial_s / 60:.0f} min "
            f"+ practice + {self.n_trials // self.break_every} breaks",
        ]
        return "\n".join(lines)
