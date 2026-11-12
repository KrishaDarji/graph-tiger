# Evaluation protocols of generative sequential recommenders (Amazon 2014 Beauty 5-core)

Purpose: make apples-to-apples comparisons with TIGER, RPG, ACERec, ActionPiece, LETTER, GRID, HSTU.
Everything below was checked against paper text or official/credible code on 2026-11-11.
"UNVERIFIED" means it could not be confirmed from the paper text or code that was read.
Local clones used for line references (not part of the repo): scratchpad `repos/` (RPG_KDD2025 @7dcf95c, action_piece @ae8e61a,
GRID @2fe3475, LETTER @8d0154e, TIGER (XiaoLongtaoo) @f1e30af, RQ-VAE-Recommender (EdoardoBotta) @957d32b,
generative-recommenders (HSTU) @9dd1e89). Paper PDFs were text-extracted with pdftotext; table cells were cross-checked where
extraction scrambled them.

---------------------------------------------------------------------------------------------------

## 0. Headline findings

1. No generative-recommender code or paper that we checked masks seen (history) items at test time. This holds for RPG
   (official), ActionPiece (official), GRID, LETTER, XiaoLongtaoo/TIGER, EdoardoBotta/RQ-VAE-Recommender. ACERec has no public code
   in the paper text (UNVERIFIED for code); it says only "full candidate ranking".
2. The TIGER paper's baseline rows (SASRec, S3-Rec, HGN, Caser, GRU4Rec, BERT4Rec, FDSA) were copied from the S3-Rec repo's
   all-item-ranking table, and that repo's code DOES mask seen items (see 1.4). The numbers match to 4 digits. So in the standard
   TIGER/RPG tables, the baselines were evaluated with seen-item masking and the generative models were (as far as code shows)
   evaluated without it. Direction of the bias: masking can only help a model, so published generative-vs-baseline gaps are, if
   anything, conservative. The magnitude is UNVERIFIED (no paper found that measures it).
3. HSTU's official research code masks seen items by default (`filter_invalid_ids=True`), but HSTU's own paper does not evaluate
   on Beauty (ML-1M, ML-20M, Amazon Books). The Beauty HSTU numbers in RPG and ActionPiece are re-implementations
   (HSTU backbone on OPQ semantic IDs), not the HSTU paper protocol.
4. On `reviews_Beauty_5.json.gz` there are zero repeated (user, item) pairs (verified locally: 198,502 reviews, 22,363 users,
   12,101 items, 0 duplicate pairs, target never appears earlier in the user's history). So masking history can never remove the
   ground truth on this dataset; it can only remove wrong candidates from the top-K. Mixing masked (stage 1) and unmasked
   (stage 2) numbers is therefore a one-directional inflation of stage 1.
5. 9,719 of 22,363 users (43.5 percent) have the last two reviews with the same `unixReviewTime`. Which item is "test" and which is
   "validation" for them depends on tie-breaking. RPG, ActionPiece and the TIGER reproductions use a stable sort on time (python
   `list.sort`), i.e. file order within a user. Any other tie-break (e.g. sorting by (time, asin), or P5's pre-built
   `sequential_data.txt`) gives different splits (UNVERIFIED how P5's file breaks ties).

---------------------------------------------------------------------------------------------------

## 1. TIGER (Rajput et al., NeurIPS 2023, arXiv 2305.05065) and its reproductions

### 1.1 What the paper states (and does not)
- Metrics: Recall@K and NDCG@K with K in {5, 10}. Section 4 "Evaluation Metrics". No other metric in the main text.
- Data: Amazon 2014 Beauty / Sports / Toys. Appendix C: "filter out users with less than 5 reviews", sequences sorted by timestamp,
  "leave-one-out strategy ... the last item is used for testing, the item before the last is used for validation, and the rest
  for training. During training, we limit the number of items in a user's history to 20." Table 6: Beauty 22,363 users, 12,101
  items, mean sequence length 8.87, median 6.
- Training schedule: Section 4 "Sequence-to-Sequence Model Implementation Details": 4+4 layers, d_model 128, ~13M params, 200k steps
  (Beauty, Sports), 100k (Toys), batch 256, lr 0.01 for 10k steps then inverse sqrt decay; user-ID hashed to 2000 tokens.
- Headline Beauty row (Table 1): R@5 0.0454, N@5 0.0321, R@10 0.0648, N@10 0.0384. Appendix Table 9 (3 seeds, mean +- stderr): Beauty
  0.0441, 0.0309, 0.0642, 0.0374 (R@5, N@5, R@10, N@10).
- NOT stated in the paper text: seen-item masking (UNVERIFIED), full ranking vs sampled negatives (UNVERIFIED; the generative decoder
  generates over the whole item space and invalid IDs are discussed in Appendix, so sampled negatives are not applicable in practice),
  beam size used for Table 1 (UNVERIFIED; Fig. 6 shows invalid-ID rate for K up to 20), how the validation checkpoint was chosen
  (UNVERIFIED; only "trained for 200k steps"), whether invalid generated IDs are dropped before Recall@K (UNVERIFIED; Appendix
  says one "can increase the beam size and filter the invalid IDs"), and how training examples are formed from a sequence
  (all prefixes vs only the last item; UNVERIFIED in the paper).
- Appendix D documents that P5's preprocessing leaks (item IDs remapped in order of first appearance, then split) and that TIGER
  re-ran P5 with random IDs; useful when comparing with any P5-derived number.
- Cold-start protocol (Section 4.2 "Cold-Start Recommendation"): "remove 5% of test items from the training data split" and train the
  RQ-VAE and the seq2seq on the training split only; unseen items matched by (c1,c2,c3) prefix; hyperparameter epsilon caps the
  share of unseen items in top-K. Recommendation-diversity: entropy of predicted category distribution (Table 3).

### 1.2 Where the baselines in TIGER Table 1 come from
- Paper text (Section 4.1): "results for all baselines, except P5, are taken from the publicly accessible results made available
  by Zhou et al. [44]" with footnote 3 = https://github.com/aHuiWang/CIKM2020-S3Rec (commit 49a890d at check time).
- That repo's README has two result tables. The paper's own table uses 99 sampled negatives (`*_sample.txt`); the second table
  (`all_rank.PNG`) ranks the ground truth against ALL items. TIGER's Beauty baseline numbers equal the all-ranking table, for
  example SASRec HR@5 0.0387 / NDCG@5 0.0249 / HR@10 0.0605 / NDCG@10 0.0318, S3-Rec 0.0387 / 0.0244 / 0.0647 / 0.0327,
  HGN 0.0325 / 0.0206 / 0.0512 / 0.0266, Caser 0.0205 / 0.0131 / 0.0347 / 0.0176 (read from the PNG and matched to TIGER Table 1
  as re-printed in RPG Table 2).
- The all-ranking code masks seen items: `trainers.py:267`:
  `rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0`
  and `run_finetune_full.py:81` sets `train_matrix = valid_rating_matrix` (items `[:-2]` of each user, `utils.py:98-114`) for validation,
  `run_finetune_full.py:130` switches to `test_rating_matrix` (items `[:-1]`, `utils.py:116-132`) for test. Masked scores are set to 0,
  not -inf (a quirk: a masked item can still outrank items whose logit is negative; effect UNVERIFIED). Max sequence length 50
  (`run_finetune_full.py:36`), checkpoint chosen by validation NDCG@20 (`run_finetune_full.py:123`).

### 1.3 XiaoLongtaoo/TIGER (unofficial PyTorch + HF T5)
- Split (`data/process.ipynb`, cell "Split data into training..."): per user, `train = seq[:-2]`, `val = seq[:-1]`, `test = seq`; each saved
  as (history = seq[:-1] of that list, target = last of that list). Sorting is a stable sort on `unixReviewTime`.
  File: `reviews_Beauty_5.json.gz` (5-core), items re-indexed from 1.
- Training examples (`model/dataset.py:25-34`): sliding prefixes of the train sequence: for i in 1..len-1, history = seq[:i], target = seq[i].
  History is left-padded / truncated to `max_len` = 20 (`dataset.py:53-69`, `main.py:204`).
- Evaluation (`model/main.py:152-176`): T5 `generate` with `num_beams = beam_size` (default 30, `main.py:213`),
  `num_return_sequences = beam_size`; first token dropped; target matches if the whole 4-token tuple equals a beam
  (`calculate_pos_index`, `main.py:97-123`). Recall@K = hit in first K beams; NDCG@K = 1/log2(rank+1) (`main.py:125-133`).
  topk_list default [5, 10, 20] (`main.py:212`). No seen-item masking anywhere in the file. Invalid (non-existent) IDs are simply misses.
  Per-batch means are averaged (`main.py:169-175`), so a short last batch is over-weighted (minor).
- Checkpoint selection (`main.py:275-298`): validation NDCG@20; test is evaluated only when validation improves; early stop patience 10.
  Test numbers are therefore not used for selection (good), but note NDCG@20 is the selection metric, not NDCG@10.
- Reported in its README: Beauty R@5 0.0392, R@10 0.0594, N@5 0.0257 (vs paper 0.0454 / 0.0648 / 0.0321).

### 1.4 EdoardoBotta/RQ-VAE-Recommender (most popular reproduction)
- Data: P5's pre-processed Amazon files (`data/amazon.py:30-31`, Google Drive `P5_data.zip`; `sequential_data.txt`), i.e. P5 ID order, which the TIGER
  Appendix D warns about (UNVERIFIED whether this repo randomises IDs; item IDs come from P5 `datamaps.json`).
- Split (`data/amazon.py:65-92`): train history = `items[:-2]`, "eval" target = `items[-2]`, "test" target = `items[-1]` with history
  `items[-(L+1):-1]`, L = 20 (`data/processed.py:32-36`). BUT `SeqData(is_train=False)` loads the "test" split
  (`processed.py:112`) and `train_decoder.py:97-107` uses it as the eval loader, so the periodic eval during training is on TEST, not
  validation. There is no checkpoint selection on validation: the model is saved at the end (`train_decoder.py:248-263`, `save_model_every`).
  Also with `train_data_subsample=True` (default, `train_decoder.py:55`) training windows are drawn from `history + future` = `items[:-1]`, which
  includes the validation item (`processed.py:158`). Do not treat its numbers as a clean leave-one-out. (Defaults: `train_decoder.py:55`.)
- Metrics (`evaluate/metrics.py`): `h@{1,5,10}` = hit rate, and `ndcg` computed over the whole beam of `top_k_for_generation` = 10
  (`decoder_amazon.gin:27`), not per K. So its NDCG is NDCG@10 only.
- Decoding (`modules/model.py:299-392`): "sampling-based beam search" (multinomial 64 candidates per step), invalid prefixes masked
  with -inf; non-deterministic across runs. No seen-item masking.

### 1.5 GRID (Snap, Ju et al., arXiv 2507.22224; github.com/snap-research/GRID)
- Paper, Section 4 "Setup": 5-core Beauty / Sports / Toys; "last item per user for test, second-to-last for validation, rest for
  training"; metrics Recall@K and NDCG@K, K in {5, 10} "on the test set, using the checkpoint with best validation Recall@10";
  5 seeds averaged; sliding-window training (Table 6: Beauty R@5 0.0396 with sliding window vs 0.0279 without).
  The paper is internally inconsistent: the same paragraph says early stopping uses NDCG@10, and the code monitors
  `val/recall@5` (`configs/experiment/tiger_train_flat.yaml:247`, patience 10 at :262).
- Code: `src/components/eval_metrics.py:262-314` `SIDRetrievalEvaluator` scores a hit when a generated SID tuple equals the label; Recall
  is hits / min(#relevant, K) (= hit rate for one target, `:120-137`), NDCG uses ideal DCG (`:87-117`). top_k_list [5, 10]
  (`tiger_train_flat.yaml:327-335`). No history masking in `tiger_generation_model.py` (searched for history/seen/filter; the only
  filter is `should_check_prefix`, default False at `:476`, i.e. unconstrained beam search by default, matching paper Table 8).
  `top_k_for_generation` = 10 beams (`:469`, `tiger_inference_flat.yaml:54`).
- Different evaluator class `RetrievalEvaluator` (`:156-259`) ranks the target against `num_negatives = 500` sampled negatives
  (`should_sample_negatives_from_vocab = True`). It is NOT the one wired to TIGER (`SIDRetrievalEvaluator` is), but do not
  confuse the two when reading numbers.
- Data: P5's pre-processed Amazon data (README link). `sequence_length: 120` tokens (`tiger_train_flat.yaml:5`) with 4 tokens per item
  is presumably 30 items (UNVERIFIED; not stated).
- Beauty numbers (RK-Means, Flan-T5-XL, Table 1, order R@5/R@10/N@5/N@10): 0.0422 / 0.0639 / 0.0277 / 0.0347; RQ-VAE 0.0404 / 0.0593 /
  0.0268 / 0.0329; TIGER-style encoder-decoder with RK-Means (Table 5) 0.0396 / 0.0597 / 0.0264 / 0.0328.

---------------------------------------------------------------------------------------------------

## 2. RPG (Hou et al., KDD 2025, arXiv 2506.05781; github.com/facebookresearch/RPG_KDD2025 @7dcf95c)

All paths relative to repo root.

### 2.1 Data and split
- Data: `reviews_{category}_5.json.gz` from SNAP (5-core) (`genrec/datasets/AmazonReviews2014/dataset.py:98`); items grouped by user and stably
  sorted by `unixReviewTime` (`:139-159`, sort at `:157`). IDs remapped from 1, 0 = PAD (`:161-191`). Text for items from `meta_*.json.gz`
  (title, price, brand, feature, categories, description, `:304-310`).
- Leave-one-out (`genrec/dataset.py:104-127`): test = full sequence (`:117-118`), val = `seq[:-1]` (`:119-121`), train = `seq[:-2]` (`:122-124`).
  Config `split: leave_one_out` (`genrec/default.yaml:71`).
- Max sequence length 50 (`genrec/default.yaml:41`, `max_item_seq_len`).
- Test input = last 50 items before the target, which INCLUDES the validation item (`genrec/models/RPG/tokenizer.py:383-393`: `item_seq[-(max+1):]`
  of the full sequence). Validation input = last 50 items of `seq[:-1]` before the validation target.

### 2.2 Training samples (augmentation)
- `genrec/models/RPG/tokenizer.py:344-382`: for a train sequence `seq[:-2]`, the first window `seq[:51]` is used with teacher forcing at EVERY position
  (`_tokenize_first_n_items`, `:295-317`: labels = next item at each position, i.e. all prefixes in one pass). If len > 51, additional sliding windows
  `seq[i:i+51]` for i = 1 .. len-52 are added with only the LAST position labelled (`_tokenize_later_items`, `:319-342`; loop `:369-375`).
  So every train position is a target exactly once (all prefixes; no duplicate prefixes).
- OPQ codebooks are fitted only on items that occur in training sequences (`_get_items_for_training`, `:158-176`; `index.train(sent_embs[train_mask])`, `:202`),
  then every item (including val/test-only items) is encoded (`index.add(sent_embs)`, `:203`). This is the right leakage hygiene for the quantiser.

### 2.3 Metrics code
- `genrec/evaluator.py:10-61`. Cutoffs from config `topk: [5,10]`, `metrics: [ndcg, recall]`, `val_metric: ndcg@10` (`genrec/default.yaml:21-23`).
  `calculate_pos_index` (`:22-38`) walks the ranked prediction list; the target index is the first j with `pred_j == label` (a hit at most once).
  `recall_at_k` = hit in top-k (`:40-41`); `ndcg_at_k` = 1/log2(rank+1), one relevant item so IDCG = 1 (`:43-48`).
- Metrics are averaged over all examples (`genrec/trainer.py:186-190`, `torch.cat(...).mean()`), correct for ragged last batches.
- No seen-item masking: exhaustive path `genrec/models/RPG/model.py:302-309` scores all `n_items-1` items and returns `topk(...).indices + 1`; graph path
  `:229-276` never consults the user's history. Searched `evaluator.py`, `trainer.py`, `model.py` for any history/mask logic: none.
- Ties: nothing explicit. `torch.topk` breaks ties arbitrarily. Two items with identical 32-digit OPQ codes would tie, which is rare. Padding id 0 is never a candidate
  (`item_id2tokens[1:]`, `:306`).
- Target index: item ids are 1-based; predictions are `index + 1` (`:308`) and compared to the label item id (`calculate_pos_index`).

### 2.4 Checkpointing and the exhaustive-vs-graph subtlety
- Training loop `genrec/trainer.py:61-146`: epochs 150 (`default.yaml:16`), `eval_interval: 1`, early-stop patience 20 (`:19`); best checkpoint chosen by
  validation `ndcg@10` (`trainer.py:130-140`); test is evaluated once with the best checkpoint (`genrec/pipeline.py:106-126`).
- Validation during training uses EXHAUSTIVE scoring (`generate_w_decoding_graph = False` initially, `model.py:88`).
  Test uses GRAPH-CONSTRAINED DECODING (`pipeline.py:119-121` sets it True before `evaluate`). So RPG test numbers are
  approximate retrieval over the item graph, validation numbers are exact. They can differ in either direction per example; typically
  slightly lower recall for the approximate version (UNVERIFIED effect size here; the paper's Fig. 6 studies b, k, q sensitivity).
- Graph decoding (`model.py:144-276`): item-item similarity = mean over the m digits of `0.5*(cos(token_emb_a, token_emb_b)+1)` between each item pair's
  tokens at that digit (`build_ii_sim_mat`, `:144-218`). `n_edges` = number of nearest neighbours kept per item (`build_adjacency_list`, `:220-221`, `topk(sim, k=n_edges)`).
  `num_beams` = beam size b: b random start items (`torch.randint(1, n_items, (B, b))`, `:238-243`; note randint samples WITH replacement and is
  unseeded per call, so test results vary slightly between calls); `propagation_steps` = q iterations of: expand each beam item to its
  `n_edges` neighbours, score each neighbour by the mean of the predicted token log-probs over its m digits, keep the top-`num_beams`
  (`:250-271`). `topk` is sorted descending, so the returned list is ranked (`:269-276`). Final top-K for K <= num_beams.
  `n_visited_items` is reported as an efficiency number.
- Defaults `default.yaml:62-65`: temperature 0.07, num_beams 50, n_edges 50, propagation_steps 3. README Beauty reproduction (`README.md:47-58`):
  lr 0.01, temperature 0.03, n_codebook 32, num_beams 20, n_edges 200, propagation_steps 3. The paper (Section 3.1 and Appendix B Table 6) states beam size 10
  and tuned edges in {50,100,500}, steps in {2,3,5}; Table 6's Beauty column extracted as 32 digits, lr 0.01, temperature 0.03, beam 10, edges 100, steps 3
  (the table text extraction is scrambled for some cells), which differs from the README's Beauty command (beam 20, edges 200). Treat the README as what the
  released code reproduces and the paper table as what the paper reports; the difference is UNVERIFIED in effect.
- Seed `rand_seed: 2024` (`default.yaml:7`); `reproducibility: True`.

### 2.4b Paper text on evaluation (Section 3.1)
"We adopt the widely used leave-last-out evaluation protocol ... Recall@ and NDCG@ ... K in {5,10}, following Rajput et al. Model checkpoints with the best
performance on the validation set are used for evaluation on the test set." Baselines: "use most of the baseline results from Rajput et al., while implementing other
baselines mainly using RecBole" (so SASRec/S3-Rec etc. rows are the S3-Rec all-rank numbers, with seen-item masking, section 1.2). Paper says "maximum of 150 steps"
(should be epochs per code). Beauty headline row (Table 2, text-embedding-3-large): R@5 0.0550, N@5 0.0381, R@10 0.0809, N@10 0.0464. With sentence-t5-base (Table 4):
N@10 0.0429 (RPG 16-64 digits) and TIGER N@10 0.0384.

### 2.5 RecBole detail relevant to baselines
RecBole's full-sort evaluation for sequential models does NOT mask history: `recbole/data/dataloader/general_dataloader.py` builds `uid2history_item` only
`if not self.is_sequential` (collate returns `None` for sequential models), and `trainer.py:_full_sort_batch_eval` only masks the pad id and `history_index` if
not None (checked on master, raw GitHub files fetched 2026-11-11; commit hash not recorded). So baselines re-run in RecBole by RPG/ActionPiece/ACERec for
datasets other than the S3-Rec-derived Beauty/Sports/Toys rows are evaluated WITHOUT masking, whereas the S3-Rec-derived rows are WITH masking.
This inconsistency between rows of the same table is in the published papers.

---------------------------------------------------------------------------------------------------

## 3. Other papers

### 3.1 ACERec (arXiv 2602.13573; no public code link found in the paper text, UNVERIFIED if released)
- Section 4.1 (v2, 2 Aug 2026): leave-last-out (last test, second-to-last validation, rest train); "All models are evaluated under full candidate ranking using Recall@K
  and NDCG@K (K in {5,10})"; results averaged over five seeds (Table 1 caption). v1 wording: "ground-truth item is ranked against the entire item set", checkpoint
  = highest validation NDCG@10.
- Appendix A.3: 5-core; sequences truncated to the 50 most recent interactions for training. Appendix A.4: baselines HGN, SASRec, S3-Rec, TIGER on
  Sports/Beauty/Toys are copied from Rajput et al.; others reproduced with official code or RecBole; one unified encoder sentence-t5-base for all semantic-ID methods; OPQ via FAISS.
  Seen-item filtering: not mentioned (UNVERIFIED).
- Beauty rows (R@5 / N@5 / R@10 / N@10), v2 Table 1: ACERec 0.0610 / 0.0433 / 0.0856 / 0.0513; RPG 0.0512 / 0.0355 / 0.0757 / 0.0436 (their rerun with
  sentence-t5-base, not RPG's published 0.0550 / 0.0381 / 0.0809 / 0.0464); ActionPiece 0.0484 / 0.0322 / 0.0762 / 0.0412 (their rerun; ActionPiece's own paper reports 0.0511 / 0.0340 / 0.0775 / 0.0424);
  TIGER 0.0454 / 0.0321 / 0.0648 / 0.0384 (copied). v1 (Feb 2026) Beauty: ACERec 0.0591 / 0.0421 / 0.0841 / 0.0501. The project README quotes ACERec 0.0841 which is the v1 value; v2 is 0.0856.
  Always cite the arXiv version.
- Other parts of the paper: sparsity analysis (test items partitioned by sparsity, Section 4 end, and Appendix C.4 on cold start; section number UNVERIFIED), latency (Appendix B.2, "average speedup of 2.2x").

### 3.2 ActionPiece (Hou et al., ICML 2025, arXiv 2502.13581; github.com/google-deepmind/action_piece @ae8e61a)
- Paper Section 4.1: leave-last-out; Recall@K and NDCG@K, K in {5,10} (Table 4); "Model checkpoints with the best performance on the validation set"; 5 seeds, mean reported.
  Appendix H: users with < 5 reviews dropped; "truncate action sequences to a maximum length of 20 for Generative methods ... For ID-based and Feature+ID baselines, we set the maximum
  length to 50"; TIGER baseline "beam size of 50"; early stopping on validation NDCG@10, patience 20, up to 200 epochs; seeds 2024-2028. Baselines BERT4Rec, SASRec, FDSA, S3-Rec, TIGER, LMIndexer
  "taken directly from existing papers" for Sports and Beauty.
- Code: `genrec/evaluator.py:20-120` same structure as RPG's (`topk [5,10,20,50]`, `metrics [ndcg, recall, err]`, `val_metric ndcg@10` in `genrec/default.yaml:36-38`); the `err` metric is the fraction of illegal generated IDs
  in the top-k, and an illegal ID simply counts as a miss. No history masking. Train: every prefix of `seq[:-2]` with a window of 20 items (`genrec/models/ActionPiece/tokenizer.py:348-364`);
  val/test: last 20 items before the target (`:365-366`). `num_beams` 50 and `n_inference_ensemble` 5 (`genrec/models/ActionPiece/config.yaml:33-34`), `max_item_seq_len` 20 (`:38`).
- Beauty (Table 4): ActionPiece 0.0511 / 0.0340 / 0.0775 / 0.0424 (R@5 / N@5 / R@10 / N@10). The same table reports HSTU 0.0469 / 0.0314 / 0.0704 / 0.0389, identical to the HSTU row in RPG Table 2.

### 3.3 LETTER (Wang et al., CIKM 2024, arXiv 2405.07314; github.com/HonghuiBao2000/LETTER @8d0154e)
- Paper Section 4.1: Instruments, Beauty, Yelp, 5-core; leave-one-out; "restrict the number of items in a user's history to 20"; metrics R@K and N@K, K in {5,10}.
  Their TIGER is a re-implementation ("official implementations have not been released") and for LC-Rec a LLaMA-7B LoRA fine-tune.
- Code `LETTER-TIGER/`: train examples from `items[:-2]` prefixes (`data.py:154-165`), valid target `items[-2]` (`:173-183`), test target `items[-1]` with history `items[:-1]` (`:191-202`),
  `max_his_len 20` (`utils.py:31`), `num_beams 20` (`utils.py:92`), metrics default hit@1,5,10 and ndcg@5,10 (`utils.py:99`); `hit_k`, `ndcg_k` in `evaluate.py:59-76`.
  Beams are re-sorted by sequence score and compared to the target string (`evaluate.py:3-47`). `--filter_items` (default True, `utils.py:84`) removes generated strings that are not valid items
  (sets their score to -1000) -- it is invalid-ID filtering, not seen-item filtering. Trie-constrained decoding is used (`generation_trie.py`, `test.py:53-62`).
- Their `Beauty.inter.json` has 22,363 users, 12,101 items, 198,502 interactions (counted locally) = identical to the standard 5-core.
- Beauty (Table 1, order R@5 / R@10 / N@5 / N@10 as printed): TIGER 0.0395 / 0.0610 / 0.0262 / 0.0331; LETTER-TIGER 0.0431 / 0.0672 / 0.0286 / 0.0364; SASRec 0.0380 / 0.0588 / 0.0246 / 0.0313
  (note: their SASRec differs from the 0.0387 / 0.0605 in TIGER Table 1, so LETTER's baselines are their own runs). Their TIGER (0.0395) is a different number than the TIGER paper (0.0454), another reminder
  that "TIGER" rows are not interchangeable.

### 3.4 HSTU (Zhai et al., ICML 2024, arXiv 2402.17152; github.com/facebookresearch/generative-recommenders @9dd1e89)
- Paper Section 4.1.1 / Table 4: MovieLens-1M, ML-20M, Amazon Books (NOT Beauty). "We report Hit Rate@K and NDCG@K over the entire corpus"; Table 4 lists HR@10, HR@50, HR@200, NDCG@10, NDCG@200.
  Training uses sampled softmax (128 negatives ML, 512 Books) but evaluation is over the full corpus.
- Code `generative_recommenders/research/data/eval.py`: metrics ndcg@{1,10,50,100,200}, hr@{1,10,50,100,200,500,1000}, mrr (`:164-198`); `get_eval_state` comment "Exhaustively eval all items (incl. seen ids)" (`:53`),
  but `eval_metrics_v2_from_tensors(..., filter_invalid_ids=True)` (`:82`) passes `invalid_ids = seq_features.past_ids` to the top-k module (`:133-139`), which masks them (`indexing/candidate_index.py:111-152`).
  The docstring says this is "consistent with papers like SASRec and TDM but may not be fair in practice" (`eval.py:87-92`). So HSTU's own protocol is MASKED full ranking; train
  uses `ignore_last_n=1` and eval the whole sequence (`data/reco_dataset.py:35-92`, no separate validation holdout in the research code), max sequence length 50 for Books
  (`configs/amzn-books/hstu-sampled-softmax-n512-final.gin:12`).
- Beauty HSTU rows in other papers are NOT the HSTU protocol: RPG says "we use the same 4-digit OPQ-tokenized semantic IDs" for HSTU; ActionPiece uses the same item features.
  Taking those rows as "HSTU" on Beauty is OK only as "HSTU backbone under the TIGER/RPG protocol" (unmasked, as far as their code shows).

---------------------------------------------------------------------------------------------------

## 4. Which Beauty numbers are comparable

Common ground (all verified): 5-core Beauty (22,363 users / 12,101 items), chronological per-user sequence, last = test, second-to-last = validation, Recall@K and NDCG@K with K in {5,10}
(single relevant item, so Recall@K = HR@K), full vocabulary (no sampled negatives) for the generative models.

Table (R@5 / N@5 / R@10 / N@10). "Masked" = seen items removed at test time according to code we read.

| Source | Model | R@5 | N@5 | R@10 | N@10 | max hist | masked? | seeds / selection | notes |
|---|---|---|---|---|---|---|---|---|---|
| TIGER paper T1 | TIGER | 0.0454 | 0.0321 | 0.0648 | 0.0384 | 20 | unstated | single run (T9: 3 seeds 0.0441/0.0309/0.0642/0.0374) | Sentence-T5-base, no public code |
| TIGER paper T1 (from S3-Rec) | SASRec | 0.0387 | 0.0249 | 0.0605 | 0.0318 | 50 | YES (code) | val NDCG@20 | copied from S3-Rec all-rank table |
| RPG paper T2 | RPG | 0.0550 | 0.0381 | 0.0809 | 0.0464 | 50 | no (code) | best val NDCG@10, test w/ graph decoding | text-embedding-3-large; with sentence-t5-base N@10 = 0.0429 (T4) |
| RPG paper T2 | HSTU (OPQ SIDs) | 0.0469 | 0.0314 | 0.0704 | 0.0389 | unstated | UNVERIFIED | | same row in ActionPiece T4 |
| ActionPiece paper T4 | ActionPiece | 0.0511 | 0.0340 | 0.0775 | 0.0424 | 20 | no (code) | 5 seeds, val NDCG@10 | sentence-t5-base |
| ACERec v2 T1 | ACERec | 0.0610 | 0.0433 | 0.0856 | 0.0513 | 50 | UNVERIFIED | 5 seeds | sentence-t5-base; v1: 0.0591/0.0421/0.0841/0.0501 |
| ACERec v2 T1 | RPG (their rerun) | 0.0512 | 0.0355 | 0.0757 | 0.0436 | 50 | UNVERIFIED | 5 seeds | fair-encoder comparison with RPG |
| GRID paper T1 | RK-Means GR | 0.0422 | 0.0277 | 0.0639 | 0.0347 | ~30 items? (UNVERIFIED) | no (code) | 5 seeds, val R@10 (paper) / R@5 (code) | Flan-T5-XL embeddings |
| LETTER paper T1 | TIGER (re-impl.) | 0.0395 | 0.0262 | 0.0610 | 0.0331 | 20 | no (code) | | own preprocessing code, beam 20 |

Comparable pairs: TIGER, RPG, ActionPiece, ACERec, GRID are all "unmasked, full-catalog, leave-last-out, R/N@5,10", so differences are real modelling differences plus these confounders:
(a) text encoder (OpenAI 3-large for RPG headline; sentence-t5-base for TIGER/ActionPiece/ACERec; Flan-T5-XL for GRID);
(b) max history 20 vs 50;
(c) checkpoint metric (NDCG@10 vs Recall@10 vs Recall@5 vs NDCG@20);
(d) seeds (single run vs 3 or 5);
(e) beam size and decoding (graph-constrained approximate vs beam search vs exhaustive).
The ID-based baseline rows (SASRec, S3-Rec, HGN, ...) are NOT comparable to the unmasked rows on a like-for-like basis (masked, history 50). ACERec's reruns of RPG/ActionPiece (their own settings) are
comparable to ACERec; RPG's published number is comparable to ACERec only after accounting for the encoder.
HSTU's native numbers (ML-1M, ML-20M, Books, HR@10/50/200) are not comparable to anything on Beauty. LETTER's Beauty tables use their own pipeline and re-implemented TIGER.

---------------------------------------------------------------------------------------------------

## 5. Recommended canonical protocol (what TIGER and RPG actually do, and honest)

Primary (headline) protocol, to be called "TIGER/RPG-style, unmasked":
1. Data: `reviews_Beauty_5.json.gz` (22,363 / 12,101). Per-user chronological sort with a STABLE sort on `unixReviewTime` (keep file order for ties; document this, since 43.5% of users
   have tied last two timestamps). Items 1-indexed, 0 = PAD.
2. Split: test target = last item, validation target = second-to-last, train = everything before. Test history = all items before the target including the validation item; validation history excludes
   the test and validation items.
3. Max history: 50 (RPG/ACERec) as the primary, with a 20-item run reported separately for TIGER-comparability. State which is used in every table.
4. Training samples: all prefixes of the train sequence (RPG `tokenizer.py:344-382` style), never the validation or test target.
5. Candidates: ALL catalog items, no masking of history items, no sampled negatives. A hit is an exact item match. Recall@K = HR@K (one target), NDCG@K with IDCG = 1.
   Report K in {5, 10} (comparison with papers) plus K = 20, 50 and MRR as extra columns.
6. Ties: use the pessimistic (or expected-rank) tie handling for exhaustive scorers (the repo's `evalkit.pessimistic_rank` is more conservative than RPG's `torch.topk`). For generated beams, a
   beam list is a ranking; invalid IDs count as misses and are reported (ActionPiece `err@K`).
7. Checkpoint: pick on validation NDCG@10 (RPG/ActionPiece/ACERec v1) once, then evaluate the test split once. Never touch test for selection (EdoardoBotta's repo evaluates on test during training; avoid).
   Report mean +- std over >= 3 seeds (5 preferred; TIGER Table 9 used 3, GRID/ActionPiece/ACERec 5).
8. Secondary (honest) protocol, reported alongside: same checkpoint, same split, but with the user's history items masked (-inf) from the candidates (HSTU, S3-Rec/SASRec convention).
   On this dataset the target is never in the history (0 repeats), so masking cannot hurt a model; the gap between the two columns is the effect size. Mask only items in the INPUT
   history (for test: all items up to and including the validation item), not the target. Always label the column. This is the only way to compare against the S3-Rec-derived SASRec rows honestly.
9. Run the same evaluation function for stage 1 and stage 2 (single module, `src/evalkit.py` already has filtered and unfiltered rank). The stage-1 vs stage-2 inconsistency is solved by reporting both
   columns for every model and never mixing them in one comparison.
10. For RPG-style models report both exhaustive scoring and graph-constrained decoding, since the paper's test numbers use the latter while model selection uses the former.

### 5.1 Additional metrics for an industry-oriented evaluation, and how to compute them without leakage
All per-user metrics below are computed on the test split with the checkpoint chosen by validation; popularity/frequency statistics use TRAIN interactions only.

| Metric | Definition | Leakage-safe computation |
|---|---|---|
| HR@K, MRR | HR = Recall@K here; MRR = mean 1/rank (rank over full catalog, capped at K or untruncated, state which) | same ranked list as Recall |
| Bootstrap CI / paired test | 95% CI from user-level bootstrap (2,000 resamples), paired across models on the same users; across seeds report mean +- std | resample users, never interactions |
| Item coverage@K | |union of top-K lists over test users| / |catalog| | catalog = all 12,101 items, list per user from the same ranker; for beam models count only valid items |
| Catalog diversity / exposure concentration | Gini or entropy of item exposure counts over test users' top-K lists; category entropy per user (intra-list distinctness by Amazon category) | measure diversity in a space the model did NOT train on (Amazon categories), not in the embedding space it consumed, or ILD is circular |
| Popularity bias | mean train-popularity of recommended items (ARP); share of top-K slots filled by head items; "popularity lift" = ARP of recs / ARP of test targets | popularity from train interactions only (`[:-2]`), bucket items head/mid/tail by train count (e.g. top 20% / next 30% / bottom 50% of interactions) |
| Long-tail recall | Recall@K and NDCG@K restricted to test users whose target is a tail item (bucket from train counts) | bucket defined on train; report n users per bucket |
| Cold-start-item recall | Recall@K on test targets with zero train interactions. Under plain leave-one-out only 33 items (0.27%) and 138 test targets (0.62%) are train-cold on Beauty (computed locally), too few to measure. Use the TIGER protocol (Section 4.2): hold out 5% of items entirely from training (drop their interactions from train sequences, keep as test targets) | for held-out items fit the quantiser (RQ-VAE / OPQ / RK-Means) on train items only, encode held-out items by text alone, and give LightGCN no embedding for them (or the gated zero). Report seen-vs-unseen split as RPG's `evaluate_cold_start` (`trainer.py:252-309`) intended (`item2group` mapping not in repo: UNVERIFIED how groups are defined) |
| Invalid-ID rate | share of beam outputs that are not catalog items (ActionPiece `err@K`; TIGER Fig. 6) | beam outputs only |
| Latency | p50 / p95 / p99 per-request wall time at batch 1, plus throughput (users/s) at a fixed batch, same hardware | warm up, `torch.cuda.synchronize()` before/after timing, include encoder + decoding + item lookup + graph construction amortisation (RPG builds the decoding graph on first call, `model.py:293-296`; report it separately), exclude data loading and metric code |
| Scaling with catalog size | latency/memory vs number of (dummy) items, as in RPG Fig. 3 | same model, grow item pool only |
| Memory | parameter count (embedding vs non-embedding, `n_parameters`), peak GPU memory (`torch.cuda.max_memory_allocated`) during inference, storage of item tables / decoding graph | report fixed batch size |
| Training cost | GPU-hours to best validation epoch, epochs/steps | log wall clock |

### 5.2 Pitfalls (each with the rule we should follow)
- Test items in LightGCN training or embeddings: build the user-item graph from train interactions `seq[:-2]` only (not `[:-1]`, which adds the validation item, and never the test item). When producing the
  final test run it is standard to still use only `[:-2]` for training; if you retrain on train+validation, state it as a different protocol. Also fit the gate, PCA, text-embedding normalisers and quantiser codebooks on train
  items only (RPG does this for OPQ, `tokenizer.py:158-203`, but fits PCA on all items, `:278-282`: PCA on text embeddings uses no interactions, so it is not interaction leakage, but it is transductive on item text; note it).
- Choosing the checkpoint on test: use validation only. XiaoLongtaoo's loop logs test when validation improves (acceptable); EdoardoBotta's loop evaluates "eval" = test (do not copy).
- Validation item in test history: standard leave-one-out says YES for test (history includes the validation item), and RPG, ActionPiece, LETTER and TIGER repos do this. Do not train on the validation item
  and also use it as history in a way the baseline does not (EdoardoBotta's subsampling trains on it).
- Ties: 43.5% of Beauty users have identical timestamps for their last two reviews; keep a stable sort; document it; consider a robustness run with a different tie-break.
  Score ties between items (identical semantic IDs) need pessimistic or random tie-breaking, not index order.
- Duplicate items in history: none on this dataset (verified), so no dedupe needed. For other datasets, mask rule (inputs only) and repeat-consumption must be decided up front.
- Sampled metrics: never mix 99-negative numbers (S3-Rec paper table, GRID's `RetrievalEvaluator`, LETTER-style `sample_num`) with full-ranking numbers. Krichene and Rendle (KDD 2020, doi 10.1145/3394486.3403226) show sampled metrics can
  invert model rankings; Dallmann et al. (RecSys 2021, arXiv 2107.13045) confirm it for sequential models.
- Global-time leakage: leave-one-out splits each user independently, so a user's test item may appear in other users' training sequences; Ji et al. (ACM TOIS 2023, arXiv 2010.11060) study this leakage, and 2607.21101 re-evaluates
  TIGER under an absolute-time split and reports a large drop (share of test targets that are unseen items "over 0.94 on Beauty and Sports" under temporal splitting; exact TIGER metric drop not read). It is out of scope for paper-comparable numbers but should be a footnote or an extra table.
- Semantic-ID collisions: TIGER appends a unique 4th code; GRID Table 7 shows random-choice vs de-dup are close (Beauty R@5 0.0381 vs 0.0396). State the collision rate.
- Beam non-determinism: RPG graph decoding starts from random items (`model.py:238`), EdoardoBotta's decoder samples candidates: seed them and report seed variance of the evaluation itself.
- Version drift: ACERec v1 vs v2 differ (Beauty R@10 0.0841 vs 0.0856); ActionPiece's own number vs ACERec's rerun differ. Cite version and table.

### 5.3 Effect size of seen-item masking
- No paper was found that measures the effect on generative recommenders on Beauty (searched arXiv and web 2026-11-11). UNVERIFIED.
- Bound: on this dataset no (user, item) pair repeats, so the ground truth is never removed by masking. Masking therefore cannot decrease Recall@K or NDCG@K for any model; it can only increase them, by an amount
  equal to how often the model ranks history items above the target in its top-K. Models that explicitly learn recency (SASRec-style) are plausibly affected more than constrained generative decoders, but this is a hypothesis.
- Cheap local measurement (recommended; no new training): for each saved checkpoint, compute test metrics twice (masked / unmasked) with the same code path and report the difference per model. The repo's `src/evalkit.py:52-56`
  already computes a filtered rank; report the unfiltered rank in the same table.

---------------------------------------------------------------------------------------------------

## 6. Not verified / open items
- TIGER paper: seen-item masking, beam size, checkpoint rule, how train examples are built, handling of invalid IDs in Table 1 (all UNVERIFIED from the paper text; no official code).
- ACERec: seen-item masking and decoding details (no code found; UNVERIFIED). Appendix B inference algorithm not checked line by line.
- HSTU numbers on Beauty in RPG/ActionPiece: how HSTU was run (masking, history length) is UNVERIFIED; RPG says it used OPQ semantic IDs.
- GRID: items per sequence implied by `sequence_length: 120` (UNVERIFIED), exact training-window construction in `NextKTokenMasking` and `collate_with_sid_causal_duplicate` (not read), paper vs code checkpoint metric mismatch.
- P5 `sequential_data.txt` ordering for tied timestamps and ID order (UNVERIFIED).
- RPG paper Table 6 Beauty edges/beam (extracted text scrambled) versus README command; effect of difference UNVERIFIED.
- RPG exhaustive vs graph-decoding gap on Beauty: not measured here.
- RecBole behaviour pinned to master at fetch time; commit hash not recorded.
- Magnitude of seen-item masking on generative or ID-based models on Beauty: no source found.
- Numbers for LETTER/GRID in the table are read from text-extracted tables; GRID order is R@5, R@10, N@5, N@10 as printed; double-check against the PDFs before quoting in slides.

## 7. Source list
- TIGER: arXiv 2305.05065 (Sections 4, 4.1, 4.2, Appendix C, D, E, Tables 1, 6, 9).
- RPG: arXiv 2506.05781 (Sections 2.3, 3.1, 3.5.2, Appendix B, Tables 2, 4, 6); code https://github.com/facebookresearch/RPG_KDD2025 (files listed in section 2).
- ACERec: arXiv 2602.13573 v1 and v2 (Section 4.1, Appendix A.3, A.4, Tables 1, 2).
- HSTU: arXiv 2402.17152 (Section 4.1.1, Table 4); code https://github.com/facebookresearch/generative-recommenders (`generative_recommenders/research/data/eval.py`, `indexing/candidate_index.py`).
- ActionPiece: arXiv 2502.13581 (Sections 4.1, 4.2, Appendix F, H, Table 4); code https://github.com/google-deepmind/action_piece.
- LETTER: arXiv 2405.07314 (Section 4.1, Table 1); code https://github.com/HonghuiBao2000/LETTER.
- GRID: arXiv 2507.22224 (Section 4 setup, Tables 1, 5-8); code https://github.com/snap-research/GRID.
- XiaoLongtaoo/TIGER https://github.com/XiaoLongtaoo/TIGER; EdoardoBotta/RQ-VAE-Recommender https://github.com/EdoardoBotta/RQ-VAE-Recommender.
- S3-Rec (source of TIGER's baseline numbers): https://github.com/aHuiWang/CIKM2020-S3Rec (`trainers.py:267`, `run_finetune_full.py:81,130`, `utils.py:98-132`, `all_rank.PNG`).
- RecBole: https://github.com/RUCAIBox/RecBole (`recbole/data/dataloader/general_dataloader.py`, `recbole/trainer/trainer.py`).
- Evaluation methodology: Krichene and Rendle, KDD 2020 (doi 10.1145/3394486.3403226); Dallmann, Zoller, Hotho, RecSys 2021, arXiv 2107.13045; Ji, Sun, Zhang, Li, arXiv 2010.11060; arXiv 2607.21101 (temporal cold-start
  re-evaluation of TIGER); arXiv 2609.24430 (semantic-ID reproducibility study; uses a global temporal split on other datasets, not Beauty).
- Local dataset facts: computed from `data/tiger_5core/raw/reviews_Beauty_5.json.gz` (read-only): 198,502 reviews, 22,363 users, 12,101 items, 0 duplicate (user,item) pairs, 9,719 users with equal timestamps on the last two reviews,
  33 items with zero train interactions under leave-one-out (138 test targets), median sequence length 6, 153 users with more than 50 reviews.
