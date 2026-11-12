# Cold start (new items)

Result in one line: for items never seen in training, a simple content-plus-graph nearest-neighbour rule ranks them better than any of our generative models, and none of the generative models puts them in the global top 10.

## Protocol (ours; TIGER's cold-start experiment uses a different, partly unspecified setup, so numbers are not comparable with it)

- Beauty 5-core, 5% of items (605, fixed random choice) never appear in any training sequence or graph. Everything else is unchanged.
- 1,228 test users have such an item as their target.
- Three views, all from the same scores (`src/evalkit.py`):
  1. **Global**: rank among all 12,101 items. This is the real situation, since the system does not know which items are new.
  2. **Cold pool**: rank among the 605 unseen items only. Measures whether the model can tell which new item fits a user, with chance Recall@10 = 0.0165.
  3. **Warm pool**: warm targets ranked among warm items only.
- Models never see a held-out item as a target or a negative. Validation and test targets are untouched.
- Unseen items get a vector in the graph slot predicted from their text (average of the 50 nearest warm items in text space, weighted; chosen over ridge regression on held-out warm items: cosine 0.455 vs 0.361). The cold items' true graph vectors are never used.

## Results (seed 2024, test)

| Method | Cold pool R@5 | R@10 | N@10 | Global cold R@10 | Global cold R@50 | Warm pool R@10 |
|---|---:|---:|---:|---:|---:|---:|
| Random | 0.0098 | 0.0171 | 0.0088 | 0.0008 (chance) | 0.004 (chance) | 0.0009 |
| Popularity (training counts) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000 | 0.0113 |
| Content nearest neighbour (last 3 items) | 0.1661 | 0.2256 | 0.1347 | n/a | n/a | 0.0426 |
| Graph nearest neighbour, imputed vectors (last item) | 0.1547 | 0.2492 | 0.1463 | n/a | n/a | 0.0630 |
| Content + graph nearest neighbour | **0.1963** | **0.2720** | **0.1623** | n/a | n/a | 0.0704 |
| Plain RPG-style model | 0.1230 | 0.1865 | 0.1002 | 0.0008 | 0.0187 | 0.0735 |
| Ours, full score | 0.1645 | 0.2288 | 0.1335 | 0.0008 | 0.0285 | 0.0937 |
| Ours, dense head only | 0.1515 | 0.2557 | 0.1322 | n/a | n/a | n/a |
| Ours, full score + score offset for unseen items (picked on validation) | n/a | n/a | n/a | 0.0049 | 0.0888 | 0.0935 |
| Ours, text-to-graph bridge trained jointly | n/a | 0.2117 | 0.1212 | 0.0008 | 0.0195 | 0.0924 |

- Not every cell was measured for every method; n/a means it was not run.
- **Ranking new items against each other**: our dense head (R@10 0.2557) is about 13% above content-only nearest neighbour and about equal to graph nearest neighbour, and the plain generative model (0.1865) is clearly worse than all three kNN rules. Combining content and graph kNN is the best at 0.2720.
- **Ranking them against everything**: all generative models leave unseen items at Global R@10 below 0.5%. A score offset for unseen items, picked on validation, raises Global R@50 from 0.029 to 0.089 at no cost to overall accuracy (overall R@10 0.0887 vs 0.0885), but Recall@10 for those users stays at 0.5%.
- A dense-head-only score for unseen items with an offset (tried with a wide grid) was chosen away by validation: any exposure given to unseen items costs more warm-user accuracy than it gains on the 5% cold targets.
- Joint training of the text-to-graph bridge (with random graph dropout) did not beat plain imputation by nearest neighbours.

## What to deploy

New items need their own path until they have interactions: keep a content-plus-graph nearest-neighbour candidate list for items with no or few interactions and reserve a fixed share of the list for it. The generative model is the right ranker for items with history. We did not find a single-model fix.
