# Figure 1 Layer-2 panel — caption draft

**Layer 2 learns multiscale templates of temporal context.** Layer 2 receives
only the excitatory rate vector `E(t)` from layer 1; layer 1 is read-only. A
bank of leaky traces with logarithmically spaced time constants represents
recent activity at several temporal scales. The outer product of current
activity and the flattened trace bank forms the directional context map
`D = E s_flat^T`; same-channel pairs are removed. Each downstream unit carries
one non-negative mask `M_k` with the same shape as `D` and responds according
to `y_k = relu(<M_k,D>)`.

Learning is local and competitive. Plasticity is gated when the norm of `D`
exceeds a fixed fraction of its running peak. The normalized context map is
compared with every mask by cosine similarity; only the best-matching mask
receives an instar update toward the current pattern, whereas all masks decay
at every time step. Frequently selected masks therefore sharpen and survive;
unused masks fade. The number of committed units is an outcome of the stream,
not a preset category count. The illustration is schematic and generic over
the number of channels, timescales and available units.
