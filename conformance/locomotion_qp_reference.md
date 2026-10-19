# Convex one-step translational control reference

This source-free Rust implementation solves two independent 1D constrained
quadratic objectives for position tracking, velocity tracking and acceleration
effort. The acceleration control is bounded by a box, and only the first
control is applied. It is deterministic for identical f64 inputs and exposes
physical units.

**Not an optimizer acceptance claim:** no multi-horizon QP solver, coupling,
velocity-norm constraint, heading/turning-radius approximation, fatigue,
collision constraints, temporal warm-start, 22-agent scheduling or per-frame
native engine integration is implemented by this candidate. It is a simple
baseline against which a real solver should be benchmarked. Linear constraints
cannot prove exact Euclidean speed/contact safety.
