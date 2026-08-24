# Benzene Pair Experimental Record

## Scope

This note records the current C, D, E, and G conclusions for the benzene pair study. Final Test MAE is the primary metric where available. The original C sweep retains relative RMSE because Final Test MAE was not reported. The G conclusions use CPU results from the first three complete paired seeds.

## C, D, and E

1. C17 remains the primary approximately 20,000 parameter MAE reference. C20 is a complementary alternating A and B reference and has not shown a consistent advantage. E311 is the primary approximately 15,000 parameter efficiency reference, nearly matching C17 while using 25.1 percent fewer parameters.

2. Full raw and mixed invariant conditioning should remain available unless it is the explicit ablation target. Repeated raw covariant access can be reduced, and expensive raw mixed covariant output branches can be pruned, when mixed invariants are preserved and the saved capacity is reassigned to the Gate.

3. Preserve declared same TypeKey covariant flow. Extra additive residual branches and deeper Gate trunks showed no reliable gain. Signed, unbounded identity coefficient output remains the most reliable default.

4. Later B refinement is promising. Gate width and selective higher order paths remain topology dependent, so C, D, and E do not establish a universal optimum.

## G

1. Generic dual source covariants were the strongest tested factor. Their pooled geometric mean Test MAE was 40.7 percent lower. The best configuration was `C on + STF@a1 on + STF@out off`.

2. The STF benefit was concentrated at `a1`. `STF@out` showed no stable gain and can be treated as redundant under the current protocol.

3. Legacy and direct carrier shortcuts were not required under the G protocol. Disabling deep raw shortcuts improved geometric mean Test MAE by about 9.1 percent across the three complete seeds. This result is specific to the G carrier shortcuts and does not overturn the C, D, and E evidence for declared same TypeKey covariant flow.

4. These claims remain limited by only three complete paired seeds. Generic pairs increased causally active parameters from about 14,900 to about 36,100, or roughly 2.42 times, so parameter efficiency is not established. The paired result summary did not include a Gate audit analysis, so invariant importance remains unresolved.
