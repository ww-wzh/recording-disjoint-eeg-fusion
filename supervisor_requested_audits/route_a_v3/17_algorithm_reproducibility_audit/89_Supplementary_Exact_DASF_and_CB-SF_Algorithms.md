# Exact DASF and CB-SF algorithms

## Scope and analysis status

DASF and CB-SF were evaluated as secondary exploratory gates in the corrected Route A v3 methodological audit. Their constants were frozen before the v3 rerun but were inherited from development conducted after inspection of earlier analyses. They were not preregistered or externally validated. Neither method optimizes CVaR, imposes a formal risk constraint, or provides a safety, no-regret, or online-deployment guarantee.

## Core fused branch

The fused branch is a class-balanced logistic stacker of neural, random-forest and Extra Trees class-1 logits. The stacker is trained only from inner out-of-fold predictions. Probabilities are clipped to [1e-6, 1-1e-6] before the logit transform. Each window receives inverse-recording-size weight, normalized to mean one. Five seeds are combined by a per-class median at each window and then renormalized; they are ensemble members, not inferential replicates.

## DASF covariance shift

Let mean(C_S) and mean(C_T) be the window-mean regularized covariance matrices in the source reference and unlabeled target batches. DASF uses

`d_cov = ||mean(C_T)-mean(C_S)||_F / (0.5[||mean(C_T)||_F+||mean(C_S)||_F])`.

If the denominator is non-finite or no larger than 1e-12, the shift is non-finite and DASF falls back to the neural branch. The effective hard-selection margin is `clip(0.02 + 0.2*d_cov, 0, 0.15)`.

## CB-SF gate diagnostics

| Order | Diagnostic | Exact definition |
|---:|---|---|
| 1 | Robust feature shift | median_j clip(z_j,0,10), z_j=abs(mean_T x_j-mean_S x_j)/std_S(x_j) |
| 2 | Branch disagreement | mean_i abs(p_NN,i(class 1)-p_fuse,i(class 1)) |
| 3 | Neural entropy | mean_i[-sum_k p_NN,ik log(p_NN,ik)] |
| 4 | Fused entropy | mean_i[-sum_k p_fuse,ik log(p_fuse,ik)] |
| 5 | Neural confidence | mean_i max_k p_NN,ik |

CB-SF feature shift is not the DASF covariance shift. It is the median capped standardized difference across the 272 flat features.

## Main constants

| Component | Parameter | Value | Status |
|---|---|---:|---|
| DASF | base margin | 0.02 | Inherited exploratory heuristic; frozen before v3 rerun, not externally validated. |
| DASF | shift coefficient | 0.2 | Inherited exploratory heuristic; frozen before v3 rerun, not externally validated. |
| DASF | maximum margin | 0.15 | Inherited exploratory heuristic; frozen before v3 rerun, not externally validated. |
| CB-SF | catastrophe threshold | 0.05 | Five-percentage-point exploratory definition; not a clinical or engineering safety limit. |
| CB-SF | risk penalty | 0.05 | Inherited exploratory heuristic; no formal risk constraint or guarantee. |
| CB-SF | temperature | 0.02 | Inherited exploratory heuristic; controls numerical softness only. |
| CB-SF | maximum fused weight | 0.8 | Inherited exploratory heuristic; keeps at least 0.2 neural contribution but is not a risk guarantee. |
| CB-SF robust scaling | feature z cap / OOD threshold | 5.0 / 10.0 | Post-hoc robustness rule, disclosed as exploratory. |
| Probability processing | stacker logit clip / entropy and aggregation clip | 1e-6 / 1e-12 | Numerical safeguards. |

The complete estimator parameter list, including scikit-learn defaults, is provided in the accompanying CSV and machine-readable JSON files.

## Nested data flow

- **Core fused branch, Every outer evaluation cell:** Base-model OOF probabilities from recording-disjoint inner folds; stacker uses only OOF logits with inverse-recording-window sample weights. Target use: Final base models generate neural, RF and Extra Trees probabilities for the untouched target recordings.
- **DASF, Bidirectional cross-task:** For each source participant, two recording-disjoint folds over four source recordings; OOF gain is calculated on source levels r1-r3. Target use: Covariance shift uses all windows from the same participant's three unlabeled target recordings r1-r3.
- **DASF, Nested LOSO:** Within each outer training population, inner participant-level OOF predictions estimate fused-minus-neural validation gain. Target use: Covariance shift uses all four unlabeled recordings of the held-out participant.
- **CB-SF, Bidirectional cross-task:** Leave-one-participant-out gate training over the other 12 participants and both directions (24 participant-direction rows). Benefit labels come from source-task OOF accuracy only. Target use: All windows from the three unlabeled target recordings jointly determine one diagnostic vector and one weight.
- **CB-SF, Arithmetic nested LOSO:** For each held-out participant, 14 inner participant OOF rows train the gate; the outer target participant is absent. Target use: All windows from all four unlabeled target recordings jointly determine one diagnostic vector and one weight.
- **CB-SF, Stroop nested LOSO:** For each held-out participant, 12 inner participant OOF rows train the gate; the outer target participant is absent. Target use: All windows from all four unlabeled target recordings jointly determine one diagnostic vector and one weight.

## Target-batch composition

| Setting | Gate cells | Training rows per target gate | Unlabeled target recordings |
|---|---:|---:|---:|
| cross_task: two directions | 26 | 24 | 3 |
| loso: arithmetic | 15 | 14 | 4 |
| loso: stroop | 13 | 12 | 4 |

In cross-task transfer, natural recordings were removed because the Arithmetic and Stroop natural files are exact same-participant duplicates. The gate therefore uses three target recordings (low, medium, and high). LOSO uses four target recordings. All recordings in a target cell are accessed jointly without labels. Consequently, this is transductive batch inference. Changing target-batch size or latent class composition can change feature means, probability summaries, and the gate weight. No invariance to such changes was established.

## Algorithm S1. Frozen core probabilities

1. In each outer evaluation cell, create recording-disjoint inner folds.
2. For each of five seeds, fit the neural, random-forest, and Extra Trees branches on each inner-training fold and predict its held-out recordings.
3. Clip each OOF class-1 probability to [1e-6, 1-1e-6] and convert it to a logit.
4. Fit the class-balanced logistic stacker only on these OOF logits, weighting every window by the inverse number of windows in its recording.
5. Select the neural epoch from inner recording-level log score, refit all branches on the complete outer-training data, and predict the untouched target data.
6. At each target window and method, take the per-class median over the five seed probabilities and renormalize the two classes.

## Algorithm S2. DASF hard selection

Input: OOF neural and fused probabilities, source covariances, unlabeled target covariances, and frozen target probabilities.

1. For each seed, calculate recording-level OOF neural accuracy A_NN and fused accuracy A_fuse.
2. Calculate g = A_fuse - A_NN.
3. Calculate d_cov = ||mean(C_target)-mean(C_source)||_F / {0.5[||mean(C_target)||_F+||mean(C_source)||_F]}.
4. Across the five seeds, separately take median(A_NN), median(A_fuse), and median(d_cov).
5. Calculate m = clip(0.02 + 0.2 d_cov, 0, 0.15).
6. If all values are finite and median(A_fuse)-median(A_NN) > m, select the fused branch; otherwise select the neural branch.
7. Return the selected frozen five-seed-ensemble probability exactly; do not recalibrate or alter it.

## Algorithm S3. CB-SF transductive soft fusion

Input: OOF gate-training rows, frozen neural and fused target probabilities, source features, and the complete unlabeled target batch.

1. For every gate-training and target cell, calculate five diagnostics: robust feature shift, branch disagreement, neural entropy, fused entropy, and neural confidence.
2. Define OOF benefit b = recording_accuracy_fuse - recording_accuracy_NN. Define catastrophe y_cat = 1[b <= -0.05].
3. For each gate feature, calculate its training median and IQR. If IQR<1e-8, use training standard deviation; if that is also <1e-8, use 1. Robust-scale training and target gate features.
4. Record the largest absolute, unclipped target robust z value. Clip model inputs to [-5,5].
5. Fit Ridge(alpha=1) to predict benefit. Clip the predicted benefit to the observed training-benefit range intersected with [-1,1].
6. If both catastrophe classes occur, fit class-balanced L2 logistic regression (C=1, solver=lbfgs, max_iter=2000); otherwise use the observed catastrophe prevalence.
7. If any target diagnostic is non-finite or max_abs_target_z>10, set lambda=0. Otherwise calculate s=b_hat-0.05 p_hat_cat and lambda=0.8 sigmoid[clip(s/0.02,-40,40)].
8. Apply the same lambda to every window in the target participant-direction batch: p_CB-SF=(1-lambda)p_NN+lambda p_fuse, then renormalize.
9. Aggregate windows to recording probabilities by the class-wise geometric mean and renormalize.

No target label is used in Steps 1-9. However, target diagnostics are batch averages. Their values, and therefore lambda, can change when the number or latent class mixture of target recordings changes.

