# Neutral Matched-Split Audit: Reproducible Protocol

This exploratory within-task diagnostic uses a fixed logistic learner, not the
Feature-MLP primary model. Arithmetic includes 15 participants and Stroop includes
13, with four recordings per participant (natural/low = 0; medium/high = 1).
The positive class for sensitivity, precision, F1, AUROC and PR-AUC is class 1.
Frozen neutral features are the same 272-dimensional eight-channel representation:
50-Hz notch (Q=30), full-record zero-phase 0.5-55-Hz filtering, 2,125-sample windows,
125-sample stride, 4,096-point FFT, no within-window z-score, and no legacy feature
or spectral multipliers. Cached float32 features are converted to float64.

Before each fit, retain features with training-window standard deviation
(ddof=0) > 1e-8 and apply a training-only StandardScaler. Fit LogisticRegression
with C=1, class_weight='balanced', solver='lbfgs', max_iter=4000, tol=1e-4,
fit_intercept=True, random_state=0, and L2 regularization (l1_ratio=0 in
scikit-learn 1.8). There is no hyperparameter search; audit LOSO is not described
as nested model selection. Nonfinite input and convergence failures stop the run.
Window rows are ordered by ascending global index. Numeric libraries use one thread.

The deliberately leaky random-window comparator uses StratifiedShuffleSplit
within participant-task, stratified by binary window labels, test_size=0.12,
two draws per seed, and seeds 1335, 1388, 1441, 1494 and 1547. All four recordings
must appear in both partitions; missing coverage is a failure, not a resampling
rule. A separate sensitivity comparator changes only test_size to 0.50.
Recording-disjoint evaluation uses the same five seeds, two complementary folds
per seed, and one whole recording per class in each train/test fold. Within-task
participant-disjoint evaluation uses leave-one-participant-out training on all
other participants and tests all four recordings of the held-out participant.

For each held-out recording, window probabilities are clipped to [1e-12,1],
geometrically averaged class-wise and renormalized; argmax ties select class 0.
Only test windows enter this aggregation. Recording-disjoint complementary folds
are pooled to four recordings before calculating each seed's metrics. Random-window
metrics use four test-recording predictions separately for each draw. Participant
scores average five seed metrics or ten draw metrics, respectively. LOSO gives
one four-recording score per participant. No averaging of draw probabilities,
no window-level inferential replication and no metric-driven threshold tuning
is performed. Reported secondary metrics use the same block-then-participant order.
Undefined precision is missing, not zero; valid block/participant counts are exported.

The primary endpoint is participant-macro recording balanced accuracy. Confidence
intervals use 10,000 participant-cluster percentile bootstrap draws with linear
2.5th/97.5th quantiles. The same task-specific participant index matrix is used
for all strategies and paired contrasts. Seed = the first eight SHA-256 bytes
of '137_neutral_matched_split_v1|TASK', interpreted little-endian; RNG is PCG64.
Two-sided Wilcoxon tests use Pratt zeros, normal approximation, no continuity
correction, and participant differences rounded to 12 decimals before ranking.
All-zero differences yield p=1. Six comparisons among the three main splits
(two tasks times three pairs) share one BH family. Six 50/50 sensitivity contrasts
share a separate exploratory BH family; all 12 also receive a single-family BH
sensitivity correction. Exact paired sign-flip sensitivity enumerates all signs
of nonzero differences, preserves zeros, and uses the absolute paired-sum statistic.
These families do not replace the 65 comparisons of the 14 primary methods.

The 280 observed recording-disjoint fits and 28 observed LOSO fits are reused
from the completed label-permutation audit only after read-only checksum,
training-label, split-index, feature-mask, scaler and full-request verification.
They are observed-label fits, not randomized-label fits. The two new random-window
comparators require 560 additional fits. Original audit results were known before
this follow-up was specified; this is not preregistration or independent validation.

Matched denotes the same features, learner and probability aggregation, not equal
training sizes or identical estimands. A 50/50 window split only approximately
matches holding out half the recordings because durations differ. Random-window
gaps combine temporal overlap, shared recording context, full-record preprocessing
and other split-induced effects; they do not isolate a causal overlap-only effect.
Participant-disjoint changes both the training population and target generalization
question. The result does not demonstrate CB-SF superiority or an online risk guarantee.

The output includes recording probabilities, participant scores, all twelve
contrasts, diagnostics, complete fold requests, and train/test global window indices.
The window archive also maps global rows to recording IDs. The JSON protocol and
source-code hashes record implementation provenance. These local artifacts do not
by themselves establish public availability; author review and a public release
remain separate steps.
