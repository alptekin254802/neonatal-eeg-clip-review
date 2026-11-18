# Planned and subsequent analyses

This retrospective summary describes the local analysis plan for the 39 held-out Helsinki infants. When we specified the evaluation, we knew the results from the 40 development infants and the published evaluation of the Duke-trained network on Helsinki recordings. The plan was dated 18 November 2026 and was not publicly preregistered.

| Question | Decision recorded before the held-out evaluation |
| --- | --- |
| Which intervals would be eligible? | Each expert served as first reader in turn. A target required at least 10 consecutive seconds marked by both other experts, with the entire interval outside the first reader's marks and their 30-second margins. |
| How much EEG would each method select? | Each method selected 5% of the eligible, non-overlapping 30-second clips for each infant and first reader, rounded down. The Duke-trained network ranked clips by mean score using the same weights, preprocessing and tie rule used in development. |
| What was the primary comparison? | For each infant with targets, we calculated the fraction captured across the three reader roles and subtracted the capture fraction expected under uniform selection at the same budget. We averaged the paired differences with equal weight per infant. Support for a gain required the lower bound of the paired infant-level 95% percentile bootstrap interval to exceed zero. |
| Which other outcomes were specified? | Selection by proximity to the first reader's marks was a secondary comparator. The plan also called for counts, visibility for at least 10 consecutive seconds, clips containing a target, concentration of targets or captures in individual infants, and the amount of EEG selected across all 39 infants and all reader roles. |

After the held-out results were known, we specified the enrichment ratio relative to uniform selection, the annotation categories in clips without a target, comparisons of captured and uncaptured target durations, the full leave-one-infant-out analysis of the paired difference, and pooled estimates across the two groups. The pooled estimate was descriptive. The primary decision used the held-out group alone.
