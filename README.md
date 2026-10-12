# Akrion deterministic soccer control foundation (candidate)

This is a **new, small Rust library** in an otherwise empty source repository,
implementing bounded research slices toward issues #33, #34 and #35.
It is not a completed 11v11 engine, global collision-avoidance guarantee,
production MPC, physics contact-resolution engine, or trained POMDP policy.

## Units, ownership, and pipeline

- Positions and radii: metres; velocities: m/s; acceleration: m/s²; control
  time step: seconds; vision half-angle: radians.
- Observation layer: reads omniscient `World` exactly once, produces a
  policy-facing `Observation` constrained by field of view, distance, basic
  circular occlusion, and discrete reaction latency.
- Policy boundary: accepts only observations, chooses hold / move / chase
  visible ball. Policy cannot access `World` through its API.
- Locomotion: generates strictly convex **2-variable single-tick** quadratic
  coefficients over the translational model
  `p_next=p+dt*v+0.5*dt²*a, v_next=v+dt*a`; linear inequalities conservatively
  bound each acceleration/speed component by `limit/sqrt(2)`.
- Reference solver: 96 deterministic cyclic halfspace projection passes. **It
  finds a feasible point or returns an explicit failure; it is NOT a general
  QP optimizer**. The prototype rolls this local solve forward for at most
  20 ticks, but applies only the first acceleration.
- Collision: broad-phase neighborhood reduction; for nearby, non-overlapping
  circular proxies create tangent halfspaces around the nominal relative
  position. Deliberate contact is exempted; overlap or undefined normals
  trigger safe-braking fallback. Analytical swept-circle checks report contact
  to a future deterministic physics engine rather than teleport or silently
  suppress tackles.
- Physics ownership: impulses, fouls, foot contact, ball dynamics, angular
  inertia, yaw rate, fatigue, uncertainty, actual body contact and legal
  tackle resolution have **NOT** been implemented.

The exterior of a disk is nonconvex. Tangent constraints plus one nominal
trajectory are at best *local conservative approximations*. Discrete-time
separation checks alone cannot exclude contact between steps. Other players
may deviate from constant-velocity predictions.

## Safety / fail-closed

Invalid or nonfinite inputs are rejected. Invalid or infeasible QP candidates
are not labeled feasible; the fallback applies bounded braking (no immediate
teleportation of velocity to zero). Fallback braking **does not guarantee
collision avoidance**, and contact should be handled by the physics engine.
Do not use this prototype as a safety-critical real-world controller.

## Run

```bash
cargo test
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo run --release --example control_benchmark
```

The benchmark prints toolchain/OS/cpu metadata where available, one-player
and 22-player timings and a deterministic control-bit checksum. It does not
measure allocations, external solvers, GPUs, inference, real 22-agent physics
or full collision crowding; those are remaining acceptance gates.

The deterministic scenarios cover constrained tracking, nominal separation,
initial overlap, head-on, crossing, chasing, crowding, deliberate contact,
swept tunneling, unseen ball, occlusion, latency and tactical re-planning.
The source repository had no existing motion code to migrate; future
integrations must be reviewed against the actual engine architecture.

## Next implementation gates

1. Integrate a **reviewed real QP solver** and expose warm/cold residual,
   objective, tolerance, infeasibility and iteration budgets.
2. Add angular inertia, heading and ball-at-feet constraints, deterministic
   contact/impulse resolution, hard speed-circle evidence and scenario baselines.
3. Add policy belief/history and visibility through moving occluders, learned
   policy adapter and platform-wide source/data provenance.
4. Add real 22-player solver benchmarks (p50/p95/p99, allocations,
   feasibility/error rates), deterministic match checksums and profiled
   selective vs all-agent control.
5. Obtain owner agreement on Linear DEN-863/DEN-862 and deployment requirements.

Keep issues #33–#35 open until those acceptance criteria are satisfied.
