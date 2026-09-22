# Quantile Balancing

Quantile Balancing is a way to choose mixture-of-experts routes now and calculate a better selection bias for the next batch. It uses the router's actual score distribution rather than moving every expert bias by a fixed step. The current routes use the current bias, while the next bias is derived from score margins and returned separately.

## Routing and weighting use different scores

For each token, begin with raw sigmoid router scores. Add the current expert bias only when deciding which experts enter the top set:

$$
\mathcal{T}_i = \operatorname{TopK}(s_i+b,k)
$$

The bias can help an underused expert get selected, but it must not distort the mixture weights after selection. Gather the selected raw scores and normalize them:

$$
p_{i,j} = \frac{s_{i,j}}{\sum_{r\in\mathcal{T}_i}s_{i,r}}
$$

This distinction is easy to miss. Biased scores choose the experts; raw scores decide how strongly the chosen experts contribute.

The expert load is simply the number of selected assignments received by each expert. If every token selects $k$ experts, the loads must sum to the number of tokens multiplied by $k$.

## The cutoff describes the selection boundary

For each token, sort its biased scores and take the value immediately below the selected top $k$. This is the $(k+1)$-th largest biased score, denoted by $\alpha_i$.

An expert score above this cutoff would enter the selected set, while one below it would not. The margin

$$
s_{i,j}-\alpha_i
$$

therefore describes how far expert $j$ is from the token's current selection boundary, using the expert's raw score against that token-specific cutoff.

## Use a quantile for the desired expert load

With $m$ tokens, $n$ experts, and $k$ selected experts per token, a perfectly balanced expert would receive

$$
q = \frac{mk}{n}
$$

assignments. This problem guarantees that $q$ is an integer.

For each expert, collect its margin across all tokens. Sort those margins from largest to smallest and take the $(q+1)$-th value. Negating this boundary gives a bias that would place roughly $q$ token margins above zero for that expert.

Finally, subtract the mean bias across experts. Adding the same constant to every expert score does not change top-k selection, so centering removes an irrelevant common offset and guarantees a zero-mean next bias.

## A small balancing picture

Suppose four tokens choose one of two experts. Then the target load is $q=4\times1/2=2$ selections per expert.

For expert A, imagine the four margins sorted from largest to smallest are $0.8$, $0.3$, $-0.1$, and $-0.5$. The $(q+1)$-th value is the third value, $-0.1$, so the uncentered next bias is $0.1$.

For expert B, suppose the corresponding boundary is $0.4$, giving an uncentered bias of $-0.4$. Their mean is $-0.15$. Subtracting it produces centered biases $0.25$ and $-0.25$.

The positive bias makes A easier to select in the next batch, while the negative bias makes B harder to select. The two values have zero mean, so only their relative difference matters.

## Keep current and next batch separate

The newly derived bias does not change the routes already selected in this call. It is returned for the caller to use with the next batch. Recomputing current routes with the new bias would mix two time steps and would make the returned loads inconsistent with the returned selected indices.

## Implementation order

- Add the current bias to raw router scores and select the top $k$ expert indices per token.
- Gather the selected raw scores and normalize them into mixture weights.
- Count how many selected assignments each expert receives.
- Find each token's $(k+1)$-th largest biased score as its cutoff.
- Form raw-score margins against those cutoffs.
- For each expert, take the $(q+1)$-th largest margin, negate it, and center all expert biases by subtracting their mean.
- Return selected indices, mixture weights, loads, and next bias in that order.

## Common mistakes to avoid

- **Normalizing biased scores.** Mixture weights come from selected raw scores.
- **Using the $k$-th score as cutoff.** The boundary is the first unselected score, which is the $(k+1)$-th largest.
- **Updating current routes with the next bias.** The new bias applies only to the next batch.
- **Using an average margin.** The update is determined by a specific load quantile.
- **Forgetting to center.** The returned next bias must have zero mean.
