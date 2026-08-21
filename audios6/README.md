# audios6 — six-tone syllables, five to a word

The experiment of `audios/word.py` at a larger grain.

    python -m audios6.word --sweep syllable
        word_syl{80,130,200,300,450,500,700,850,1000}ms.mp3
    python -m audios6.word --sweep tone
        word_tone{0,10,20,40,80,160}ms.mp3
    --order perm     the trajectory control
    --redraw         the floor control

A sibling directory rather than an option on the old module: almost every
constant moves, and a version that tried to be both would be readable as
neither. The shared machinery (`audios.core`, `audios.cloud`) is imported, not
copied — only the design lives here.

## What changed, and why

| | `audios` | here | reason |
|---|---|---|---|
| tones per syllable | 5 | **6** | syllable now spans 0–100 ms at the same 20 ms tone step |
| syllables per word | 3 | **5** | thirty figure tones per word |
| figure channels | 15 | **30** | half the old pool, so the pool widens |
| pool | 61 ch, 250 Hz–8 kHz | **67 ch, 178 Hz–8 kHz** | keeps 37 non-figure channels |
| word period | 1600 ms | **2000 ms** | five syllables at the speech rate is a 900 ms word |
| word jitter | ±240 ms | **±280 ms** | still ~14% of the period, still whole 40 ms blocks |

The tone sweep is unchanged in structure — one syllable, six tones now instead
of five — and still tops out at 160 ms, because the tone step is the
syllable's own frozen shape rather than a way of stringing syllables together.

## The frequency layout

The one thing that had to be solved rather than scaled. Thirty figure tones
must not collide in the channel grid, which needs

    syl_step * dj  !=  tone_spacing * dk

for every pair in range. Five semitones between tones against three between
syllables needs `dj = 5`, impossible with five syllables. It is also the most
compact arrangement that manages it: **thirty distinct channels inside
thirty-seven semitones**, where the obvious alternatives need forty-three.
The figure sits at −18…+19 st (354 Hz–3.0 kHz), with cloud above and below it.

## The cloud control, unchanged

Everything that made the background honest carries over, and is re-measured
here rather than assumed:

    concurrency          6.00 +- 0.03, deepest dip 0.8 dB, 0.0% of the time
    channel use          43-44 tones over 67 channels
      figure channels    43-44, mean 43.3   (30 channels)
      other              43-44, mean 43.3   (37 channels)
    per quarter          9-12 per channel (quarter totals 724-726)

That is: the cloud sounds the figure's own channels, every channel is used
equally in total *and* through the stream, the total number of tones sounding
never moves, and the word onset is jittered on the cloud's 40 ms grid so the
figure is findable neither by frequency, by rhythm, nor by envelope.

See `../audios/README.md` for why each of those is necessary — the reasoning
is identical and is not repeated here.
