# Post-hoc preprocessing sensitivity analysis

## Scope

This reviewer-requested analysis was specified after inspection of the primary Route A v3 results and is therefore exploratory. It does not modify the frozen main predictions and no preprocessing variant may replace the reference pipeline because it yields a higher descriptive result.

Five preprocessing definitions were compared with an identical fixed StandardScaler and class-balanced logistic-regression learner. The corrected cohorts were used: Arithmetic S01-S15, Stroop S01-S13, and within-participant bidirectional cross-task transfer for S01-S13. Cross-task source cells contained raw levels 0-3 and target cells contained levels 1-3; the duplicated target natural recording was excluded. Recording-level balanced accuracy was primary. Percentile intervals used 10,000 participant-level bootstrap resamples, and 24 paired Wilcoxon tests were adjusted as one Benjamini-Hochberg family.

The reference pipeline used complete-window channel-wise z-scoring, a 0.03 multiplier for spectral power from 48.5 to 51.5 Hz, a 1.12 multiplier for the first four channel positions, and a joint 1.10 theta/alpha relative-power multiplier. The four variants removed window z-scoring, replaced it with complete-recording z-scoring, removed the 50-Hz spectral multiplier, or jointly neutralized the two legacy emphasis multipliers. These are implementation sensitivities rather than optimized alternatives.

## Descriptive results

| evaluation_display | condition_display | variant | n_participants | mean_recording_balanced_accuracy | balanced_accuracy_ci95_low | balanced_accuracy_ci95_high |
| --- | --- | --- | --- | --- | --- | --- |
| Within-participant cross-task transfer | Arithmetic to Stroop | Neutral legacy multipliers | 13 | 51.92% | 42.31% | 61.54% |
| Within-participant cross-task transfer | Arithmetic to Stroop | No 50-Hz attenuation | 13 | 55.77% | 44.23% | 69.23% |
| Within-participant cross-task transfer | Arithmetic to Stroop | No within-window z-score | 13 | 51.92% | 38.46% | 65.38% |
| Within-participant cross-task transfer | Arithmetic to Stroop | Recording-level z-score | 13 | 44.23% | 36.54% | 51.92% |
| Within-participant cross-task transfer | Arithmetic to Stroop | Reference preprocessing | 13 | 51.92% | 42.31% | 61.54% |
| Within-participant cross-task transfer | Stroop to Arithmetic | Neutral legacy multipliers | 13 | 40.38% | 23.08% | 59.62% |
| Within-participant cross-task transfer | Stroop to Arithmetic | No 50-Hz attenuation | 13 | 51.92% | 38.46% | 65.38% |
| Within-participant cross-task transfer | Stroop to Arithmetic | No within-window z-score | 13 | 51.92% | 40.38% | 63.46% |
| Within-participant cross-task transfer | Stroop to Arithmetic | Recording-level z-score | 13 | 65.38% | 50.00% | 78.85% |
| Within-participant cross-task transfer | Stroop to Arithmetic | Reference preprocessing | 13 | 42.31% | 25.00% | 61.54% |
| Participant-disjoint LOSO | Arithmetic | Neutral legacy multipliers | 15 | 58.33% | 53.33% | 65.00% |
| Participant-disjoint LOSO | Arithmetic | No 50-Hz attenuation | 15 | 58.33% | 50.00% | 65.00% |
| Participant-disjoint LOSO | Arithmetic | No within-window z-score | 15 | 53.33% | 46.67% | 60.00% |
| Participant-disjoint LOSO | Arithmetic | Recording-level z-score | 15 | 60.00% | 51.67% | 70.00% |
| Participant-disjoint LOSO | Arithmetic | Reference preprocessing | 15 | 58.33% | 53.33% | 65.00% |
| Participant-disjoint LOSO | Stroop | Neutral legacy multipliers | 13 | 59.62% | 50.00% | 69.23% |
| Participant-disjoint LOSO | Stroop | No 50-Hz attenuation | 13 | 61.54% | 50.00% | 73.08% |
| Participant-disjoint LOSO | Stroop | No within-window z-score | 13 | 61.54% | 51.92% | 71.15% |
| Participant-disjoint LOSO | Stroop | Recording-level z-score | 13 | 50.00% | 36.54% | 61.54% |
| Participant-disjoint LOSO | Stroop | Reference preprocessing | 13 | 59.62% | 50.00% | 69.23% |
| Within-participant recording-disjoint | Arithmetic | Neutral legacy multipliers | 15 | 54.67% | 50.33% | 59.67% |
| Within-participant recording-disjoint | Arithmetic | No 50-Hz attenuation | 15 | 55.00% | 50.33% | 60.33% |
| Within-participant recording-disjoint | Arithmetic | No within-window z-score | 15 | 59.00% | 52.00% | 65.67% |
| Within-participant recording-disjoint | Arithmetic | Recording-level z-score | 15 | 47.67% | 42.00% | 53.00% |
| Within-participant recording-disjoint | Arithmetic | Reference preprocessing | 15 | 54.67% | 50.00% | 59.67% |
| Within-participant recording-disjoint | Stroop | Neutral legacy multipliers | 13 | 52.69% | 42.69% | 62.69% |
| Within-participant recording-disjoint | Stroop | No 50-Hz attenuation | 13 | 50.77% | 43.08% | 58.46% |
| Within-participant recording-disjoint | Stroop | No within-window z-score | 13 | 47.31% | 39.62% | 55.00% |
| Within-participant recording-disjoint | Stroop | Recording-level z-score | 13 | 44.23% | 37.69% | 50.38% |
| Within-participant recording-disjoint | Stroop | Reference preprocessing | 13 | 51.54% | 42.31% | 60.77% |

## Paired differences from the reference pipeline

| evaluation_display | condition_display | variant | n_paired_participants | mean_balanced_accuracy_difference | paired_balanced_accuracy_ci95_low | paired_balanced_accuracy_ci95_high | wilcoxon_balanced_accuracy_p_raw | wilcoxon_p_bh_24_comparison_family |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Within-participant cross-task transfer | Arithmetic to Stroop | Neutral legacy multipliers | 13 | +0.00 pp | +0.00 pp | +0.00 pp | 1.0 | 1.0 |
| Within-participant cross-task transfer | Arithmetic to Stroop | No 50-Hz attenuation | 13 | +3.85 pp | -3.85 pp | +13.46 pp | 0.5323782283120907 | 0.901924578641237 |
| Within-participant cross-task transfer | Arithmetic to Stroop | No within-window z-score | 13 | +0.00 pp | -11.54 pp | +9.62 pp | 0.7460057700658305 | 0.994674360087774 |
| Within-participant cross-task transfer | Arithmetic to Stroop | Recording-level z-score | 13 | -7.69 pp | -21.15 pp | +5.77 pp | 0.2103552635998217 | 0.6310657907994651 |
| Within-participant cross-task transfer | Stroop to Arithmetic | Neutral legacy multipliers | 13 | -1.92 pp | -5.77 pp | +0.00 pp | 0.31731050786291415 | 0.6923138353372672 |
| Within-participant cross-task transfer | Stroop to Arithmetic | No 50-Hz attenuation | 13 | +9.62 pp | +0.00 pp | +19.23 pp | 0.09641736216836228 | 0.6310657907994651 |
| Within-participant cross-task transfer | Stroop to Arithmetic | No within-window z-score | 13 | +9.62 pp | -3.85 pp | +23.08 pp | 0.14692598078374663 | 0.6310657907994651 |
| Within-participant cross-task transfer | Stroop to Arithmetic | Recording-level z-score | 13 | +23.08 pp | +5.77 pp | +42.31 pp | 0.048771993285610445 | 0.6310657907994651 |
| Participant-disjoint LOSO | Arithmetic | Neutral legacy multipliers | 15 | +0.00 pp | +0.00 pp | +0.00 pp | 1.0 | 1.0 |
| Participant-disjoint LOSO | Arithmetic | No 50-Hz attenuation | 15 | +0.00 pp | -6.67 pp | +6.67 pp | 1.0 | 1.0 |
| Participant-disjoint LOSO | Arithmetic | No within-window z-score | 15 | -5.00 pp | -10.00 pp | +0.00 pp | 0.0832645166635504 | 0.6310657907994651 |
| Participant-disjoint LOSO | Arithmetic | Recording-level z-score | 15 | +1.67 pp | -6.67 pp | +10.00 pp | 0.7054569861112734 | 0.994674360087774 |
| Participant-disjoint LOSO | Stroop | Neutral legacy multipliers | 13 | +0.00 pp | +0.00 pp | +0.00 pp | 1.0 | 1.0 |
| Participant-disjoint LOSO | Stroop | No 50-Hz attenuation | 13 | +1.92 pp | -5.77 pp | +9.62 pp | 0.6547208460185769 | 0.9820812690278654 |
| Participant-disjoint LOSO | Stroop | No within-window z-score | 13 | +1.92 pp | -3.85 pp | +7.69 pp | 0.5637028616507731 | 0.901924578641237 |
| Participant-disjoint LOSO | Stroop | Recording-level z-score | 13 | -9.62 pp | -23.08 pp | +1.92 pp | 0.20433024733906624 | 0.6310657907994651 |
| Within-participant recording-disjoint | Arithmetic | Neutral legacy multipliers | 15 | +0.00 pp | +0.00 pp | +0.00 pp | 1.0 | 1.0 |
| Within-participant recording-disjoint | Arithmetic | No 50-Hz attenuation | 15 | +0.33 pp | -4.00 pp | +4.00 pp | 0.3756950400558079 | 0.7513900801116158 |
| Within-participant recording-disjoint | Arithmetic | No within-window z-score | 15 | +4.33 pp | -1.00 pp | +9.67 pp | 0.1605295351439231 | 0.6310657907994651 |
| Within-participant recording-disjoint | Arithmetic | Recording-level z-score | 15 | -7.00 pp | -14.33 pp | +0.33 pp | 0.13695337682873407 | 0.6310657907994651 |
| Within-participant recording-disjoint | Stroop | Neutral legacy multipliers | 13 | +1.15 pp | +0.00 pp | +3.46 pp | 0.31731050786291415 | 0.6923138353372672 |
| Within-participant recording-disjoint | Stroop | No 50-Hz attenuation | 13 | -0.77 pp | -5.77 pp | +4.23 pp | 0.8475568684403095 | 1.0 |
| Within-participant recording-disjoint | Stroop | No within-window z-score | 13 | -4.23 pp | -10.00 pp | +0.77 pp | 0.4138944330247134 | 0.7641127994302401 |
| Within-participant recording-disjoint | Stroop | Recording-level z-score | 13 | -7.31 pp | -18.85 pp | +4.62 pp | 0.3089692200478711 | 0.6923138353372672 |

Complete participant-level and fold-level outputs accompany this supplement. Neither zero-phase filtering nor complete-window/complete-recording standardization is a causal sample-by-sample operation; the analysis supports only offline or delayed completed-window use.
