# Layer-2 illustration design

The panel is a single left-to-right sentence rather than a stack of algorithm
boxes. Five numbered nodes establish the forward computation: layer-1 rates,
multiscale temporal traces, a directional coincidence map, a bank of learned
masks and the complete vector of unit activities. Each stage is represented by
its native visual object and one concise equation.

The learning rule forms a smaller return loop beneath `D` and `M`. Its four
steps—gate, normalize, compete, adapt plus forget—are visually subordinate to
the forward pass but remain fully explicit. This separation prevents
winner-take-all learning from being mistaken for winner-take-all output: all
unit activities are shown, while only the best-matching mask updates.

The artwork uses the manuscript palette and a white, unboxed editorial field.
Blue marks the layer-1 input, teal the temporal trace, terracotta the
directional/plasticity path and violet the layer-2 templates and responses.
Typography is fixed at final print size, never below 5 pt, and the primary
stage labels are 6.15 pt or larger. PDF and SVG remain editable vectors; the
PNG is a 600-dpi review rendering.
