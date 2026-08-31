"""The whole experiment in one object.  Nothing is decided anywhere else.

Elhilali, Ma, Micheyl, Oxenham & Shamma (2009), Neuron 61:317-329, methods
pp. 11-12 of the author manuscript.  Every number below that carries a
`# paper` comment is theirs; the rest are ours and say why.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Design:
    # --- what is being asked ------------------------------------------
    # 'replicate'  the published experiment: the A stream runs at a
    #              different tempo from the B stream (30 or 70 ms gaps) or
    #              at the same one (50 ms), and the threshold for spotting a
    #              shift of the last B tone reports whether the two were
    #              heard as one stream.
    # 'sweep'      the question the paper only simulated (Fig. 8): hold the
    #              two tempi identical and slide the whole B stream by a
    #              constant lag, from synchronous to fully alternating.
    #              Same measurement, one continuous axis.
    mode: str = "replicate"      # 'replicate' | 'sweep'

    # --- the tones ----------------------------------------------------
    f_a: float = 1000.0          # paper: A fixed at 1000 Hz
    df_st: tuple[float, ...] = (6.0, 9.0, 15.0)
                                 # paper: B at 0.5, 0.75, 1.25 octaves above
                                 # A.  15 st is wider than an octave and is
                                 # the condition that makes the point
    tone_ms: float = 100.0       # paper: 100 ms *including* both ramps, so
                                 # the steady part is 80 ms
    ramp_ms: float = 10.0        # paper: raised cosine

    # --- the sequence -------------------------------------------------
    n_precursor: int = 5         # paper: five precursor tones at each
                                 # frequency, then one target tone each
    gap_b_ms: float = 50.0       # paper: the B stream is the reference and
                                 # its gap never moves
    gap_a_ms: tuple[float, ...] = (30.0, 50.0, 70.0)
                                 # paper: 50 ms makes the two streams
                                 # synchronous with identical tempi; 30 and
                                 # 70 make the A stream run faster or slower,
                                 # which is the published way of producing
                                 # asynchrony
    b_only: bool = True          # paper's third control: A tones off, so the
                                 # listener has to hear the irregularity in
                                 # the B stream alone.  This is the floor the
                                 # asynchronous conditions have to be
                                 # compared against, not chance

    # --- the sweep ----------------------------------------------------
    # Fig. 8 simulated 75 ms tones slid from 0 % to 100 % of the way to
    # alternation and watched the second singular value of the coherence
    # matrix rise.  Nobody measured it in a listener.  75 ms tones against a
    # 150 ms period is the one length at which 100 % is exactly alternation:
    # A fills the first half of the period and B the second, with no overlap
    # and no gap.  With the paper's 100 ms tones the two would still overlap
    # 25 ms at the far end and the axis would not reach its own endpoint.
    sweep_tone_ms: float = 75.0
    sweep_gap_ms: float = 75.0   # so the period is 150 ms in both modes and
                                 # each frequency repeats at the same rate as
                                 # in the replication
    sweep_pct: tuple[float, ...] = (0.0, 10.0, 20.0, 30.0, 40.0,
                                    50.0, 65.0, 80.0, 100.0)
                                 # per cent of the way from synchronous to
                                 # alternating, as Fig. 8 plots it.  Dense
                                 # below half, because the paper's own 40 ms
                                 # (53 %) partially-overlapping sequence is
                                 # reported to already sound alternating
    sweep_df_st: tuple[float, ...] = (6.0, 15.0)
                                 # the narrow and the wide separation.  Three
                                 # would be better and costs another 40 min

    # --- level --------------------------------------------------------
    # The paper does not state a presentation level for the human
    # experiment.  70 dB SPL is what they used in the ferret and is the
    # usual choice for this stimulus.  Calibrate once and leave it.
    spl_db: float = 70.0
    rms_dbfs: float = -26.0      # per tone, before summing
    ear: str = "left"            # paper: the left earpiece of an HD 580.
                                 # Monaural, so no binaural cue can help
    rove_db: float = 0.0         # off, as in the paper.  Both intervals of a
                                 # trial are the same sequence apart from the
                                 # shift, so level cannot be the cue
    random_phase: bool = True    # starting phase of each tone, drawn per
                                 # trial and shared by the two intervals.
                                 # Not in the paper, and it costs nothing:
                                 # with a fixed phase the A and B tones
                                 # overlap the same way in every trial of a
                                 # condition and that one alignment is
                                 # available to be learned over two hours

    # --- the trial ----------------------------------------------------
    isi_ms: float = 500.0        # paper: 500 ms between the two intervals
    task: str = "2i2afc"         # paper: which interval ended asynchronously
    response_s: float = 10.0
    feedback: bool = True        # standard for an adaptive track, and the
                                 # paper's listeners had "extensive
                                 # experience with the test procedure"
    feedback_ms: float = 350.0
    iti_ms: float = 600.0

    # --- the adaptive track -------------------------------------------
    # paper: three-down one-up, which converges on 79.4 % correct.  dT is
    # divided by c after two consecutive correct answers and multiplied by c
    # after each wrong one; c starts at 4, drops to 2 at the first reversal
    # from decreasing to increasing, and to sqrt(2) two reversals later; the
    # run ends at the sixth reversal taken with the sqrt(2) step, and the
    # threshold is the geometric mean of dT at those six reversals.
    dt_start_ms: float = 20.0
    factors: tuple[float, ...] = (4.0, 2.0, math.sqrt(2.0))
    factor_after: tuple[int, ...] = (1, 3)   # reversals at which c steps down
    n_final_reversals: int = 6
    dt_min_ms: float = 0.25      # a floor the paper does not need to state:
                                 # dividing by 4 three times from 20 ms lands
                                 # at 0.3 ms, which is 15 samples at 48 kHz.
                                 # Runs that sit on the floor are counted and
                                 # reported rather than silently averaged in
    dt_max_ms: float = 40.0      # a backward shift eats into the 50 ms gap
                                 # before the target, and at 50 ms it would
                                 # abut the tone before it into one long tone.
                                 # 40 leaves 10 ms, still plainly two events,
                                 # and it is twice the largest threshold the
                                 # paper reports
    runs_per_cell: int = 4       # paper: at least four threshold
                                 # measurements per listener per condition
    max_trials: int = 120        # a runaway track is a failed run, not a
                                 # long one

    # --- practice -----------------------------------------------------
    practice_runs: int = 1        # one track at the easiest cell, discarded
    practice_df_st: float = 6.0

    # --- rendering ----------------------------------------------------
    fs: int = 48_000             # the paper used 32 kHz; 48 is the modern
                                 # default and the highest tone here is
                                 # 2378 Hz, so neither rate is near a limit
    lead_ms: float = 200.0       # silence before and after each interval, so
    tail_ms: float = 200.0       # the sequence never starts at sample zero

    # --- housekeeping -------------------------------------------------
    break_every: int = 8         # tracks, not trials
    break_min_s: float = 20.0
    show_condition: bool = False  # a listener who can see the condition will
                                  # use it
    seed: int = 20260831

    # ------------------------------------------------------------------
    def replace(self, **kw) -> "Design":
        return replace(self, **kw)

    # --- derived ------------------------------------------------------
    @property
    def n_tones(self) -> int:
        """Tones per frequency: the precursors and then the target."""
        return self.n_precursor + 1

    @property
    def tone(self) -> float:
        return self.sweep_tone_ms if self.mode == "sweep" else self.tone_ms

    @property
    def soa_b(self) -> float:
        """Onset to onset within the B stream."""
        gap = self.sweep_gap_ms if self.mode == "sweep" else self.gap_b_ms
        return self.tone + gap

    def soa_a(self, gap_ms: float) -> float:
        return self.tone + gap_ms

    def f_b(self, df_st: float) -> float:
        return self.f_a * 2.0 ** (df_st / 12.0)

    def alternation_ms(self) -> float:
        """The lag at which the B stream sits exactly between two A tones."""
        return self.soa_b / 2.0

    def lag_ms(self, pct: float) -> float:
        return self.alternation_ms() * pct / 100.0

    @property
    def conditions(self) -> list[dict]:
        """Every cell of the experiment, in one list."""
        out = []
        if self.mode == "sweep":
            for df in self.sweep_df_st:
                for pct in self.sweep_pct:
                    out.append(dict(kind="sweep", df_st=df, pct=pct,
                                    lag_ms=self.lag_ms(pct),
                                    gap_a_ms=self.sweep_gap_ms, b_only=False))
            return out
        for df in self.df_st:
            for g in self.gap_a_ms:
                out.append(dict(kind="gap", df_st=df, pct=None, lag_ms=0.0,
                                gap_a_ms=g, b_only=False))
            if self.b_only:
                out.append(dict(kind="b_only", df_st=df, pct=None,
                                lag_ms=0.0, gap_a_ms=self.gap_b_ms,
                                b_only=True))
        return out

    @property
    def n_runs(self) -> int:
        return len(self.conditions) * self.runs_per_cell

    def interval_ms(self, gap_a_ms: float, lag_ms: float = 0.0) -> float:
        """How long one interval lasts, worst case."""
        span_b = (self.n_tones - 1) * self.soa_b + self.tone + lag_ms
        span_a = (self.n_tones - 1) * self.soa_a(gap_a_ms) + self.tone
        # the two streams are aligned on their target tones, so the earlier
        # start is whichever stream reaches its target later
        return self.lead_ms + max(span_a, span_b) + self.dt_max_ms \
            + self.tail_ms

    @property
    def trial_s(self) -> float:
        w = max(self.interval_ms(g) for g in
                (self.gap_a_ms if self.mode == "replicate"
                 else (self.sweep_gap_ms,)))
        return (2 * w + self.isi_ms + self.feedback_ms + self.iti_ms) / 1000.0

    @property
    def est_trials_per_run(self) -> float:
        """A three-down one-up run to six reversals, empirically."""
        return 45.0

    @property
    def est_minutes(self) -> float:
        return (self.n_runs * self.est_trials_per_run * (self.trial_s + 1.2)
                / 60.0)

    # ------------------------------------------------------------------
    def validate(self) -> None:
        if self.mode not in ("replicate", "sweep"):
            raise ValueError(f"unknown mode {self.mode!r}")
        if self.ear not in ("left", "right", "both"):
            raise ValueError(f"unknown ear {self.ear!r}")
        if self.ramp_ms * 2 > self.tone:
            raise ValueError("the ramps are longer than the tone")
        if self.dt_start_ms >= self.dt_max_ms:
            raise ValueError("the track starts outside its own range")
        if self.mode == "sweep":
            if abs(self.sweep_tone_ms + self.sweep_gap_ms - self.soa_b) > 1e-9:
                raise ValueError("sweep period does not match")
            if max(self.sweep_pct) > 100.0:
                raise ValueError("a lag past 100 % is a lag the other way")
            # 100 % must be exact alternation, which needs tone == period/2
            if abs(self.sweep_tone_ms - self.alternation_ms()) > 1e-9:
                raise ValueError(
                    f"at 100 % the tones would overlap by "
                    f"{self.sweep_tone_ms - self.alternation_ms():.0f} ms; "
                    f"set sweep_tone_ms to {self.alternation_ms():.0f}")
        for g in self.gap_a_ms:
            if g <= 0:
                raise ValueError("a non-positive gap abuts two tones into one")
        # the shifted target must still be a separate tone from its neighbour
        if self.dt_max_ms >= min(self.gap_b_ms, self.sweep_gap_ms):
            raise ValueError(
                f"dt_max_ms {self.dt_max_ms:g} would close the "
                f"{min(self.gap_b_ms, self.sweep_gap_ms):g} ms gap before the "
                f"target and fuse it with the tone before it")

    # ------------------------------------------------------------------
    def summary(self) -> str:
        c = self.conditions
        if self.mode == "sweep":
            head = (f"sweep  {len(self.sweep_pct)} lags "
                    f"{', '.join('%g' % p for p in self.sweep_pct)} % "
                    f"x {len(self.sweep_df_st)} separations")
            tones = (f"  tones     {self.tone:.0f} ms, {self.soa_b:.0f} ms "
                     f"period, alternation at {self.alternation_ms():.0f} ms "
                     f"lag")
        else:
            head = (f"replication  {len(self.df_st)} separations "
                    f"{', '.join('%g' % d for d in self.df_st)} st "
                    f"x {len(self.gap_a_ms)} A-gaps"
                    f"{' + B-only control' if self.b_only else ''}")
            tones = (f"  tones     {self.tone:.0f} ms incl. "
                     f"{self.ramp_ms:.0f} ms ramps, B gap "
                     f"{self.gap_b_ms:.0f} ms, A gap "
                     f"{'/'.join('%g' % g for g in self.gap_a_ms)} ms")
        return "\n".join([
            head,
            f"  sequence  {self.n_precursor} precursors + 1 target per "
            f"stream, A at {self.f_a:.0f} Hz",
            tones,
            f"  track     3-down 1-up on dT, {self.dt_start_ms:.0f} ms start, "
            f"steps {'/'.join('%.3g' % f for f in self.factors)}, "
            f"{self.n_final_reversals} reversals",
            f"  session   {len(c)} cells x {self.runs_per_cell} runs = "
            f"{self.n_runs} tracks, about {self.est_minutes:.0f} min",
        ])
