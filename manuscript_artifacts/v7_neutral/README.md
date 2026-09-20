# v7 neutral evidence package

These files are the machine-readable reporting artifacts for the v7 manuscript.
They were generated from the canonical 2,660-row recording prediction file and
the neutral preprocessing protocol. Local numbered output directories are not
part of the public provenance.

The primary endpoint is participant-macro balanced accuracy. The matched split
audit uses the same learner and feature representation for random-window,
recording-disjoint, and participant-disjoint evaluation; only the split unit
changes. The random-window comparator is intentionally leaky and is retained as
an audit control, not as a valid deployment estimate.

All gate and held-out-gain analyses are exploratory. Non-significance is not
interpreted as equivalence, formal risk control, or validated online safety.
