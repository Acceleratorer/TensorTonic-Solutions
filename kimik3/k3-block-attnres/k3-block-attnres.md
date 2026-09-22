# Block Attention Residuals

Block Attention Residuals keeps the main benefit of depth attention while storing fewer sources. Instead of retaining every earlier layer output separately, it adds consecutive layer outputs into block summaries. The current layer then retrieves from the embedding, completed block sums, and possibly one partial sum from the block currently being built.

## Why blocks are used

Full Attention Residuals can retrieve from every earlier layer, but the number of stored representations grows with network depth. Block attention reduces this memory by representing a group of nearby layers with one sum.

The embedding remains its own source because it is the original input to the stack. After that, each complete group of the requested block size contributes one source. If some recent layers do not yet fill a complete block, their sum contributes one partial source.

This exercise focuses on constructing those sources correctly. Once they are built, the supplied helper performs the same normalized depth-attention read used by the full version.

## Construct completed blocks in order

For block size $m$, take the preceding layer outputs in chronological order and divide them into consecutive groups of $m$. A complete block sum is

$$
B_j = \sum_{i=jm}^{(j+1)m-1} h_i
$$

Do not average the layers. Summation is the representation defined by the method. Also do not form groups by taking every $m$-th output; each block must contain adjacent layers.

The source list begins with the embedding, followed by completed block sums from oldest to newest.

## Handle the current partial block

If the number of preceding outputs is not divisible by the block size, add one more source containing the sum of the remaining recent outputs. This source represents the portion of the current block that has already been produced.

If the count is exactly divisible by the block size, there is no remainder and therefore no partial source. Adding a zero tensor or an empty sum in this case would create an extra attention option that does not correspond to any representation.

## A concrete grouping example

Let the block size be three and suppose seven preceding layer outputs are available:

- The embedding is source zero.
- Layers 1, 2, and 3 are summed into the first completed block.
- Layers 4, 5, and 6 are summed into the second completed block.
- Layer 7 becomes the partial-block source.

The helper receives four sources in total. If only six preceding outputs were available, it would receive the embedding and two completed blocks, with no partial source.

Suppose scalar layer outputs are $1,2,3,4$ and block size is three. The first block source is $1+2+3=6$, and the partial source is $4$. Together with an embedding value of $5$, the depth sources are $[5,6,4]$. The attention helper scores and mixes these three values.

## Retrieval uses the shared full-attention rule

After source construction, the helper RMS-normalizes each source for scoring, compares it with the pseudo-query, applies softmax over the source axis, and mixes the original sources. You do not need to duplicate this logic.

The returned block-source tensor matters as much as the retrieved value. It allows the tests to verify grouping, ordering, and exact-boundary behavior directly.

## Implementation order

- Begin the source list with the embedding.
- Determine how many complete groups of the requested block size exist.
- Sum every complete consecutive group and append it in chronological order.
- If outputs remain, sum them into exactly one partial source.
- Stack the sources and pass them to the supplied retrieval helper.
- Return the retrieved value, attention weights, and source stack in that order.

## Common mistakes to avoid

- **Averaging within blocks.** Block representations are sums.
- **Dropping the embedding.** It is always the first source.
- **Adding an empty partial block.** Exact block boundaries must not create an extra source.
- **Reordering layers.** Blocks are chronological groups of consecutive outputs.
- **Returning only the attention result.** The problem also requires weights and the constructed block sources.
