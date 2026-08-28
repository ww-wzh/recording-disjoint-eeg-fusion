# Final table and figure captions

## Table 1

**Cohort and recording structure.** Window counts refer to 8.5-s windows advanced by 0.5 s. Cross-task target cells exclude the natural recording because the Arithmetic and Stroop natural files are exact within-participant duplicates. Stroop LOSO excludes S14 and S15 because their high-level files are byte-identical and provenance could not be resolved.

## Table 2

**Participant-macro recording-level balanced accuracy for all 14 methods in the three complete analysis settings.** Values are percentage means with two-sided 95% percentile participant-bootstrap intervals in parentheses. Five random seeds are ensemble members rather than statistical replicates. Descriptively highest values do not establish statistical superiority.

## Table 3

**Paired comparison of DASF and CB-SF with the prespecified Always-neural comparator.** Differences are method minus comparator in percentage points. Confidence intervals use 10,000 participant-cluster percentile resamples. Wilcoxon tests use Pratt zero handling and an approximate two-sided calculation because of zeros and ties. Both reporting-family and single-family 65-comparison Benjamini-Hochberg adjusted p values are provided. No row established superiority.

## Table 4

**Matched split audit using an identical fixed learner and hyperparameters.** Random-window splitting places highly overlapping windows from the same recording on both sides of the split. Recording-disjoint and participant-disjoint estimates use complete held-out groups.

## Table 5

**Post-hoc preprocessing sensitivity analysis.** Differences are relative to the reference preprocessing and use participant-level recording balanced accuracy. The 24 comparisons form one Benjamini-Hochberg family. These results cannot be used to select a replacement primary pipeline.

## Figure 1

**Effect of data partitioning on participant-macro recording balanced accuracy.** Points are participant-macro means and error bars are two-sided 95% percentile participant-bootstrap intervals. The annotated paired contrasts compare random-window with recording-disjoint splitting after averaging split repeats within each participant. The participant, rather than the window or split seed, is the inferential unit.

## Figure 2

**All-method results in the three complete analysis settings.** Points are participant-macro recording balanced accuracy and horizontal lines are two-sided 95% percentile participant-bootstrap intervals. The dashed vertical line marks 50% balanced accuracy. Colors identify the primary comparator, proposed exploratory gates, and raw-EEG deep baselines; gray points are other comparators. Numerical ordering is descriptive and does not imply superiority.

## Supplementary Figure S1

**Direction-specific cross-task results.** Points and intervals use participants as clusters. The two directions were added as post-hoc directional reporting and were adjusted separately from the three complete settings, with an additional all-65-family sensitivity adjustment.
