# Deviations from the pre-registration

Recorded on 2026-10-05, in time order. The pre-registration file itself is not edited.

## D1. Non-inferiority test (recorded after idea 11, before ideas 6, 7 and 2 were run)

The pre-registered non-inferiority test is a one-sided Wilcoxon signed-rank test on the
paired differences plus the margin. When most paired differences are exactly zero (binary
survival, and R@10 for nearly identical rankings), the shift turns every tie into a small
positive value. Those ties take the lowest ranks but are so numerous that the positive rank
sum dominates, so the test can reject H0 even when the new method is clearly worse than the
margin. Idea 11 shows this: survival fell by 0.19 against a margin of 0.02 and the
pre-registered test still gave p = 9e-16.

Change. For every non-inferiority hypothesis (H6, H7, H11), the pre-registered Wilcoxon
p-value is still reported, and a one-sided paired t-test on (new minus comparator plus
margin), H0: mean difference <= -margin, is added. Acceptance requires both tests (each
Holm-adjusted in the family, the t-test p entering the family in place of the Wilcoxon p
when it is larger) as well as the point-estimate rule. This can only make acceptance harder.
Superiority tests are unaffected, because there the zero differences are dropped and
not shifted.

## D2. Idea 9 base rate (descriptive, added after the gate was evaluated)

The diagnostic now also reports the share of all notes within one and two links of the
dense top 10, as a base rate for the gated share. The gate itself is unchanged.

## D3. Corpus snapshot

The live corpus gained two chunks between the baseline run and this study. All ideas use one
frozen snapshot (1,223 notes with chunks, 5,157 chunks), cached once. The dense baseline on
the snapshot reproduces the reference values exactly (R@10 0.698391, MRR 0.516824,
nDCG@10 0.537789, chunk-pack survival 0.755202).
