# Exact training protocols and baseline-comparison audit

## Common evaluation rules

All 14 methods were evaluated on the same frozen Route A v3 outer test recordings. Cross-task analysis used 13 participants in both transfer directions, Arithmetic LOSO used 15 participants, and Stroop LOSO used 13 participants. No test label was used for fitting, early stopping, stacking, or gate diagnostics. Window probabilities were aggregated by a geometric mean within recording, and participants, rather than windows or random seeds, were the statistical units. The five fixed seeds (1335, 1388, 1441, 1494, 1547) were ensemble members and were never treated as repeated observations.

The protocol was aligned with respect to outer partitions, disjoint inner partitions where model selection was required, target recordings, label mapping, probability aggregation, and statistical analysis. It was not computationally budget-matched: deterministic Riemannian methods were fitted once, fixed blends had no fitted parameters, and the raw-signal deep models used a representation-specific 0.5-45 Hz front end whereas engineered-feature and covariance methods used 0.5-55 Hz. Accordingly, the comparison is described as protocol-aligned, not as an equal-compute benchmark.

## Method roles and fit policies

| Method | Role | Input | Independent fits | Selection | Comparison status |
| --- | --- | --- | --- | --- | --- |
| Feature MLP | Primary comparator | 272 engineered features; 0.5-55 Hz | 5 | Same nested inner folds | Directly protocol-aligned |
| Window-level heterogeneous stack | Learned fusion comparator | OOF logits from Feature MLP, RF and Extra Trees | 5 | Same nested inner folds; OOF only | Protocol-aligned derived model |
| Random forest | Classical feature baseline | 272 engineered features; 0.5-55 Hz | 5 | Same nested inner folds for OOF fusion evidence | Directly protocol-aligned |
| Extra Trees | Classical feature baseline | 272 engineered features; 0.5-55 Hz | 5 | Same nested inner folds for OOF fusion evidence | Directly protocol-aligned |
| Fixed 10% blend | Secondary fixed-weight baseline | Frozen neural and heterogeneous-stack probabilities | 0 | No fitted parameters | Derived from the same frozen branches |
| Fixed 25% blend | Secondary fixed-weight baseline | Frozen neural and heterogeneous-stack probabilities | 0 | No fitted parameters | Derived from the same frozen branches |
| Equal 50% blend | Secondary fixed-weight baseline | Frozen neural and heterogeneous-stack probabilities | 0 | No fitted parameters | Derived from the same frozen branches |
| Recording-level OOF stacker | Secondary learned meta-baseline | Recording-level OOF branch logits | 1 | Source-recording OOF for cross-task; nested participant OOF for LOSO | Protocol-aligned but trained at recording level |
| DASF hard gate | Exploratory proposed selector | Frozen neural/fused probabilities plus label-free shift | 0 | OOF validation evidence only | Exploratory derived decision rule |
| CB-SF soft gate | Post-hoc exploratory proposed gate | Five target-batch diagnostics from frozen branches and features | 0 | Cross-fitted gate training; target labels excluded | Exploratory derived decision rule |
| Riemannian tangent-space logistic regression | Conventional geometric baseline | 8-channel covariance; 0.5-55 Hz | 1 | No inner model selection | Same outer test set; deterministic and differently represented |
| Riemannian MDM | Conventional geometric baseline | 8-channel covariance; 0.5-55 Hz | 1 | No inner model selection | Same outer test set; deterministic and differently represented |
| EEGNet | Raw-EEG deep baseline | 8 x 2125 raw window; 0.5-45 Hz | 5 | Same recording-/participant-disjoint nested folds | Same split/ensemble budget; representation-specific front end |
| EEG-Conformer | Raw-EEG deep baseline | 8 x 2125 raw window; 0.5-45 Hz | 5 | Same recording-/participant-disjoint nested folds | Same split/ensemble budget; representation-specific front end |

## Exact configurations

### Feature MLP

| Parameter | Value | Scope |
| --- | --- | --- |
| input dimension | 272 | all v3 settings |
| trainable parameters | 467234 | all v3 settings |
| hidden width | 256 | all v3 settings |
| residual blocks | 3 | all v3 settings |
| dropout | 0.2 | all v3 settings |
| optimizer | AdamW | all v3 settings |
| learning rate | 0.001 | all v3 settings |
| weight decay | 0.0001 | all v3 settings |
| batch size | 128 | all v3 settings |
| maximum epochs | 100 | all v3 settings |
| early-stopping patience | 15 | all v3 settings |
| loss | inverse-frequency class-weighted cross-entropy | all v3 settings |
| gradient clipping | L2 norm 1.0 | all v3 settings |
| feature scaling | StandardScaler fitted only on the corresponding training fold | all v3 settings |
| epoch selection | recording-level mean true-class log probability; final epoch is median inner-fold best epoch | all v3 settings |
### Random forest

| Parameter | Value | Scope |
| --- | --- | --- |
| n_estimators | 500 | all v3 settings |
| max_depth | 30 | all v3 settings |
| min_samples_leaf | 4 | all v3 settings |
| max_features | 0.25 | all v3 settings |
| n_jobs | -1 | all v3 settings |
| class weight | balanced | all v3 settings |
| outer-cell seed offset | +37 for RF; +74 for Extra Trees | each ensemble seed |
### Extra Trees

| Parameter | Value | Scope |
| --- | --- | --- |
| n_estimators | 500 | all v3 settings |
| max_depth | 30 | all v3 settings |
| min_samples_leaf | 4 | all v3 settings |
| max_features | 0.25 | all v3 settings |
| n_jobs | -1 | all v3 settings |
| class weight | balanced | all v3 settings |
| outer-cell seed offset | +37 for RF; +74 for Extra Trees | each ensemble seed |
### Window-level heterogeneous stack

| Parameter | Value | Scope |
| --- | --- | --- |
| inputs | class-1 logits from Feature MLP, RF and Extra Trees | each inner/outer cell |
| C | 1.0 | each inner/outer cell |
| class weight | balanced | each inner/outer cell |
| maximum iterations | 2000 | each inner/outer cell |
| training predictions | OOF only | each inner/outer cell |
| sample weight | inverse number of windows in each recording | each inner/outer cell |
| probability clipping before logit | [1e-6, 1-1e-6] | each inner/outer cell |
### EEGNet

| Parameter | Value | Scope |
| --- | --- | --- |
| D | 2 | all v3 settings |
| F1 | 8 | all v3 settings |
| F2 | 16 | all v3 settings |
| dropout | 0.5 | all v3 settings |
| pool1 | 4 | all v3 settings |
| pool2 | 8 | all v3 settings |
| temporal_kernel | 125 | all v3 settings |
| input shape | 8 x 2125 | every window |
| trainable parameters | 3834 | fixed architecture |
| band-pass | 0.5-45 Hz, fourth-order zero-phase Butterworth | each recording before windowing |
| normalization | per-window per-channel z-score | each completed 8.5-s window |
| batch size | 32 | all v3 settings |
| learning rate | 0.001 | all v3 settings |
| max epochs | 80 | all v3 settings |
| patience | 12 | all v3 settings |
| selection metric | recording-level mean log probability | all v3 settings |
| weight decay | 0.0001 | all v3 settings |
| loss | inverse-frequency class-weighted cross-entropy | each training fold |
| optimizer | AdamW | each fit |
| gradient clipping | L2 norm 1.0 | each update |
| final epoch | median inner-fold selected epoch | each outer cell and seed |
### EEG-Conformer

| Parameter | Value | Scope |
| --- | --- | --- |
| attention_heads | 4 | all v3 settings |
| dropout | 0.5 | all v3 settings |
| embedding_dim | 40 | all v3 settings |
| feedforward_dim | 160 | all v3 settings |
| pool_kernel | 75 | all v3 settings |
| pool_stride | 15 | all v3 settings |
| temporal_kernel | 25 | all v3 settings |
| transformer_layers | 2 | all v3 settings |
| input shape | 8 x 2125 | every window |
| trainable parameters | 58962 | fixed architecture |
| band-pass | 0.5-45 Hz, fourth-order zero-phase Butterworth | each recording before windowing |
| normalization | per-window per-channel z-score | each completed 8.5-s window |
| batch size | 32 | all v3 settings |
| learning rate | 0.001 | all v3 settings |
| max epochs | 80 | all v3 settings |
| patience | 12 | all v3 settings |
| selection metric | recording-level mean log probability | all v3 settings |
| weight decay | 0.0001 | all v3 settings |
| loss | inverse-frequency class-weighted cross-entropy | each training fold |
| optimizer | AdamW | each fit |
| gradient clipping | L2 norm 1.0 | each update |
| final epoch | median inner-fold selected epoch | each outer cell and seed |
### Riemannian tangent-space logistic regression

| Parameter | Value | Scope |
| --- | --- | --- |
| input | 8 x 8 regularized covariance per window | all v3 settings |
| reference fitting | outer-training windows only | each outer cell |
| logistic C | 1.0 | all v3 settings |
| logistic max iter | 4000 | all v3 settings |
| tangent metric | riemann | all v3 settings |
| class weight | balanced | all v3 settings |
| tangent scaling | StandardScaler fitted on outer training | each outer cell |
### Riemannian MDM

| Parameter | Value | Scope |
| --- | --- | --- |
| input | 8 x 8 regularized covariance per window | all v3 settings |
| reference fitting | outer-training windows only | each outer cell |
| mdm metric | riemann | all v3 settings |
### Fixed 10% blend

| Parameter | Value | Scope |
| --- | --- | --- |
| formula | (1-0.10)*P_neural + 0.10*P_heterogeneous | window probability |
| fitted parameters | 0 | all settings |
### Fixed 25% blend

| Parameter | Value | Scope |
| --- | --- | --- |
| formula | (1-0.25)*P_neural + 0.25*P_heterogeneous | window probability |
| fitted parameters | 0 | all settings |
### Equal 50% blend

| Parameter | Value | Scope |
| --- | --- | --- |
| formula | (1-0.50)*P_neural + 0.50*P_heterogeneous | window probability |
| fitted parameters | 0 | all settings |
### Recording-level OOF stacker

| Parameter | Value | Scope |
| --- | --- | --- |
| inputs | recording-level class-1 branch logits | protocol-specific |
| C | 1.0 | protocol-specific |
| class weight | balanced | protocol-specific |
| maximum iterations | 4000 | protocol-specific |
| probability clipping before logit | [1e-6, 1-1e-6] | protocol-specific |
| cross-task fit data | four source-task recording-level OOF rows in the target participant cell | protocol-specific |
| LOSO fit data | inner participant-level OOF recording predictions from outer-training participants | protocol-specific |
| target labels used | False | protocol-specific |

## Architecture sizes

The fixed implementations contained 467,234 trainable parameters for the 272-input Feature MLP, 3,834 for EEGNet, and 58,962 for EEG-Conformer. No architecture or hyperparameter sweep was conducted during the v3 rerun.

## Reproducibility environment

| Item | Value |
| --- | --- |
| Operating system | Windows-11-10.0.26200-SP0 |
| Python | 3.12.13 |
| NumPy | 2.4.4 |
| pandas | 3.0.2 |
| scikit-learn | 1.8.0 |
| SciPy | 1.17.1 |
| PyTorch | 2.9.0+cu129 |
| PyTorch CUDA runtime | 12.9 |
| cuDNN | 91002 |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| pyRiemann | 0.11 |
| Random seeds | 1335, 1388, 1441, 1494, 1547 |
| PyTorch deterministic setting | cudnn.deterministic=True; cudnn.benchmark=False |
| Seed statistical role | ensemble members only; never statistical replicates |

## Interpretation of fairness

The Feature MLP, RF, Extra Trees, EEGNet, and EEG-Conformer used five genuine seeded fits and identical outer/inner split identities. The Riemannian models are deterministic; their probabilities were copied into five seed slots solely to preserve a common file schema, so this does not constitute a five-model ensemble. Fixed blends, DASF, and CB-SF are deterministic transformations of frozen branch predictions. The recording-level stacker is a separately fitted OOF meta-baseline. These distinctions must accompany performance tables and prevent seed slots from being counted as independent evidence.
