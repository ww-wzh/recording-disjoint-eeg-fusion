# Exact preprocessing and 272-dimensional feature definition

## Scope

This supplement documents the implementation used in the corrected eight-channel exploratory audit. The retained channels, in fixed order, were Fp1, Fp2, F7, F3, Fz, F4, F8, and C2. The OpenBCI packet counter and every auxiliary column were excluded. The implementation is an offline completed-window pipeline and is not a causal sample-by-sample or single-recording online system.

## Preprocessing

1. **Fail-closed channel selection.** Detect modulo-256 packet counter in column 0; retain columns 1-8 only. Exactly eight finite EEG columns are required; the parser refuses to guess when extra columns are ambiguous.
2. **Continuous-recording band-pass.** Fourth-order Butterworth 0.5-55 Hz, applied with forward-backward sosfiltfilt before windowing. Forward-backward filtering is zero-phase but non-causal; this is an offline analysis pipeline.
3. **Overlapping windows.** 8.5 s (2125 samples) with 0.5 s (125 samples) stride. Adjacent windows share 94.12% of raw samples and must never be randomly separated across training and validation.
4. **Within-window per-channel z-score.** x_c(t) <- [x_c(t)-mean_t(x_c)] / max(std_t(x_c), 1e-6). Uses the complete 8.5-s window. It is valid for completed-window offline/batch inference, not causal sample-by-sample prediction.
5. **Hann taper and FFT.** Apply a 2125-point Hann window and zero-pad to a 4096-point real FFT. The implementation does not use a 2048-point FFT. Zero-padding refines the frequency grid but does not add physical frequency resolution.
6. **50-Hz spectral attenuation.** Multiply FFT power bins from 48.5 through 51.5 Hz by 0.03. This is not a designed time-domain notch filter. The value 0.03 is an inherited heuristic, not a learned or theoretically guaranteed constant.
7. **Fixed spectral emphasis.** Multiply all relative band powers in Fp1/Fp2/F7/F3 by 1.12; multiply theta and alpha relative powers in every channel by 1.10. The values 1.12 and 1.10 were not selected inside the v3 nested protocol and must not be presented as optimized or physiologically validated constants.
8. **Covariance output for Riemannian baselines.** C = X X^T / 2125 + 1e-4 I for each standardized 8-channel window. The covariance matrix is not part of the 272-dimensional flat feature vector. An 8-channel tangent vector has 8*9/2=36 dimensions, not 253.
9. **Raw-EEG deep-baseline preprocessing.** Fourth-order zero-phase 0.5-45 Hz band-pass followed by the same within-window per-channel z-score. Its upper cutoff differs from the 55-Hz engineered-feature path. This must be disclosed as a representation-specific preprocessing difference, not described as an identical front end.

## Feature-vector composition

| Block | Feature group | Indices | Dimension |
|---:|---|---:|---:|
| 1 | Log absolute band power | 1-48 | 48 |
| 2 | Relative band power | 49-96 | 48 |
| 3 | Spectral entropy | 97-104 | 8 |
| 4 | Theta/beta and alpha/beta ratios | 105-120 | 16 |
| 5 | Time-domain summaries | 121-152 | 32 |
| 6 | Across-channel spatial summaries | 153-172 | 20 |
| 7 | Hjorth descriptors | 173-196 | 24 |
| 8 | Physiological extras | 197-213 | 17 |
| 9 | Cross-band log summaries | 214-216 | 3 |
| 10 | Frontal workload summaries | 217-218 | 2 |
| 11 | Channel-band differential entropy | 219-266 | 48 |
| 12 | Across-channel differential entropy | 267-272 | 6 |

The blocks sum to 272 features. The separately stored 8 x 8 regularized covariance matrices were used by the Riemannian baselines and were not appended to this vector.

### S1. Log absolute band power (48 features)

Formula: `log(mean_{f in band} P_c(f) + epsilon)`

Six bands for each of eight channels; the frontal multiplier is added in log space to channels 1-4.
### S2. Relative band power (48 features)

Formula: `mean_{f in band} P_c(f) / mean_{0.5<=f<55} P_c(f)`

Six bands per channel; channels 1-4 are multiplied by 1.12 and theta/alpha entries by 1.10.
### S3. Spectral entropy (8 features)

Formula: `-sum_f q_c(f) log(q_c(f)), q_c(f)=P_c(f)/sum_f P_c(f)`

Computed per channel over the implemented 0.5-55 Hz mask.
### S4. Theta/beta and alpha/beta ratios (16 features)

Formula: `theta/(low-beta+high-beta); alpha/(low-beta+high-beta)`

Two ratios for each channel using relative powers after the fixed multipliers.
### S5. Time-domain summaries (32 features)

Formula: `mean(x); std(x); sqrt(mean(x^2)); max(x)-min(x)`

Four summaries per channel after within-window z-score; mean, standard deviation and RMS are therefore nearly fixed by construction.
### S6. Across-channel spatial summaries (20 features)

Formula: `mean_c(relative); std_c(relative); mean_c(log absolute); mean_c(entropy); std_c(entropy)`

Six values for each of the first three summaries plus two entropy summaries.
### S7. Hjorth descriptors (24 features)

Formula: `log(var(x)); sqrt(var(dx)/var(x)); sqrt(var(d2x)/var(dx))/mobility`

Activity, mobility and complexity for each channel.
### S8. Physiological extras (17 features)

Formula: `mean(abs(dx)); mean(alpha first 4)-mean(alpha last 4); (delta+theta)/(alpha+low-beta+high-beta)`

The one-dimensional alpha term is a channel-group contrast, not a standard left-right frontal alpha asymmetry measure.
### S9. Cross-band log summaries (3 features)

Formula: `log(mean_c(alpha/beta)); log(mean_c(gamma/alpha)); log(mean_c(theta/alpha))`

Three across-channel ratios.
### S10. Frontal workload summaries (2 features)

Formula: `log(mean_{c=1..4}(theta)/mean_{c=1..4}(beta)); log(mean_{c=1..4}(alpha))`

The first four ordered channels are Fp1, Fp2, F7 and F3.
### S11. Channel-band differential entropy (48 features)

Formula: `0.5 log(2*pi*e*var(x_{c,band}))`

Six FFT-mask reconstructed band signals for each of eight channels.
### S12. Across-channel differential entropy (6 features)

Formula: `mean_c[0.5 log(2*pi*e*var(x_{c,band}))]`

One channel-mean differential-entropy value per band.

## Status of fixed constants

The 50-Hz power multiplier (0.03), the first-four-channel spectral multiplier (1.12), and the theta/alpha multiplier (1.10) were inherited heuristic settings. They were not optimized within the corrected nested protocol and do not provide a formal physiological or risk guarantee. Their influence is therefore treated only in a post-hoc preprocessing sensitivity analysis.

## Deployment interpretation

Forward-backward filtering and standardization over each complete 8.5-s window use future samples relative to the beginning of that window. The present results therefore support only offline or delayed completed-window batch inference. Moreover, the selective gate uses all unlabeled recordings in a target participant-direction cell and is transductive; it is not a conventional single-recording online gate.
