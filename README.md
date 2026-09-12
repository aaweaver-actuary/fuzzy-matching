# fuzzy-matching

Reference implementation for conservative insurance-carrier entity resolution, motivated by mapping agent-entered carrier names to AM Best financial groups.

The intended flow is:

1. normalize AM Best references and user input identically;
2. resolve exact/known aliases deterministically first;
3. generate plausible candidates;
4. compute complementary similarity, information and ambiguity features;
5. fit a calibrated classifier estimating `P(top candidate is correct)`;
6. auto-accept only above a business-selected precision threshold; otherwise return candidates for review;
7. promote reviewed aliases into a versioned deterministic alias table.

## Input data contract

### AM Best/reference data

At minimum, each reference row should contain a stable group identifier and display name:

```text
group_id | group_name
---------|----------------------------------
123      | Cincinnati Financial Corporation
456      | American Family Insurance Group
```

If AM Best company/legal-entity names or authoritative aliases are available, preserve them rather than collapsing them away. A practical long-form reference table is:

```text
group_id | group_name | reference_name | reference_type
---------|------------|----------------|---------------
123      | ...        | ...            | group_name
123      | ...        | ...            | company_name
123      | ...        | ...            | known_alias
```

Every `reference_name` should receive BOTH representations:

```python
reference_conservative = normalize_conservative(reference_name)
reference_aggressive = normalize_aggressive(reference_name)
```

Fit `TokenIDF` and `CharNgramTFIDF` on the **conservatively normalized reference universe**. IDF is meant to describe how discriminating a token is among possible answers, not how common it happens to be in one agency's submissions.

If multiple legal entities roll to the same AM Best financial group, retain all useful legal names as candidate aliases but keep the group identifier as the resolution target.

### User/agency input

Preserve the original value for auditing:

```python
raw_carrier = "Cincinnati Ins. Co."
query_conservative = normalize_conservative(raw_carrier)
query_aggressive = normalize_aggressive(raw_carrier)
```

Do not preprocess user input differently from reference names. Any abbreviation expansion such as `ins -> insurance` must apply symmetrically.

For historical labeled examples, retain at least:

```text
agency_id
book_roll_id
raw_carrier
true_group_id
observation_date
```

`agency_id` matters because agencies may have persistent spelling and abbreviation conventions. Validation intended to measure generalization to new books should generally hold out entire agencies rather than randomly splitting account rows.

## Normalization philosophy

`normalize_conservative()` is the primary representation. It currently:

1. performs Unicode normalization/ASCII folding;
2. lowercases;
3. standardizes `&` to `and`;
4. removes punctuation;
5. normalizes whitespace;
6. expands a small audited dictionary of whole-token abbreviations;
7. removes only relatively safe legal-form tokens such as `inc` and `llc`.

Example:

```python
normalize_conservative("Cincinnati Ins. Co., Inc.")
# "cincinnati insurance company"
```

`normalize_aggressive()` starts with the same transform, then additionally removes generic terms such as `insurance`, `company`, `corporation`, and `group`:

```python
normalize_aggressive("Cincinnati Ins. Co., Inc.")
# "cincinnati"
```

Aggressive normalization is intentionally lossy. It is useful as additional evidence and for candidate generation. It should not automatically imply identity unless the resulting value is known to map uniquely to one group.

The abbreviation and stop-word sets in `normalization.py` are examples, not claims about optimal AM Best rules. Extend them only with reviewed domain-specific transformations. Avoid broad substring replacements; use whole-token or carefully bounded rules.

## Feature modules

### `string_metrics.py`

`rapidfuzz_ratio`, `partial_ratio`, `token_sort_ratio`, `token_set_ratio`, `wratio`, `jaro_winkler`, and `damerau_levenshtein`. Pairwise similarities return `[0, 1]`.

### `idf.py`

Information-weighted token features: `shared_idf`, `max_shared_idf`, `weighted_jaccard`, `query_idf_coverage`, `candidate_idf_coverage`, `unmatched_query_idf`, and `unmatched_candidate_idf`.

These distinguish agreement on rare identity-bearing words from agreement on generic terms such as `insurance` or `company`.

### `ngrams.py`

`CharNgramTFIDF` fits character 3-5 gram TF-IDF vectors on the normalized reference universe and provides cosine similarity. This supplies typo-tolerant signal distinct from ordinary edit distance.

### `acronyms.py`

`acronym_exact`, `acronym_similarity`, and `query_looks_like_acronym` target initials and abbreviated multi-token names.

### `ambiguity.py`

Candidate-landscape features include `margin_1_2`, `margin_1_3`, `n_within_5`, `n_within_10`, `top_n_mean`, `top_n_std`, and `score_softmax_entropy`.

## Score-scale convention

Pairwise feature functions return similarities on `[0, 1]` for modeling consistency. Candidate-landscape functions accept whatever common score scale you supply. If candidate ranking uses native RapidFuzz `[0, 100]` scores, `n_within_5` means five RapidFuzz points. If ranking scores are `[0, 1]`, use `n_within_delta(scores, 0.05)` instead.

## Example

```python
from fuzzy_matching.features import pairwise_features
from fuzzy_matching.idf import TokenIDF
from fuzzy_matching.ngrams import CharNgramTFIDF
from fuzzy_matching.normalization import normalize_conservative

raw_reference_names = [
    "Cincinnati Financial Corporation",
    "American Family Insurance Group",
    "Liberty Mutual Insurance Group",
]

reference_names = [normalize_conservative(x) for x in raw_reference_names]
idf = TokenIDF.fit(reference_names)
char_model = CharNgramTFIDF.fit(reference_names)

query = normalize_conservative("Cincinnatti Financial")
candidate = reference_names[0]

features = pairwise_features(query, candidate, idf=idf)
features["char_ngram_cosine"] = char_model.similarity(query, candidate_index=0)
```

## Training/calibration data

A useful first modeling row is one historical query after candidate generation:

```text
agency_id
raw_carrier
true_group_id
top_candidate_group_id
<top-candidate pairwise features>
<candidate-landscape features>
top_candidate_correct
```

with:

```text
top_candidate_correct = 1 if top_candidate_group_id == true_group_id else 0
```

The target is therefore the probability that the proposed top candidate is actually the intended group.

Avoid treating repeated rows from one agency/book as independent evidence. For calibration on unseen business, use agency-grouped cross-validation and consider deduplicating repeated `(agency_id, raw_carrier, true_group_id)` patterns for query-weighted evaluation.

## Development

```bash
python -m pip install -e ".[dev]"
pytest
```

Tests cover normalization symmetry, IDF behavior, typo similarity, acronym matching, character n-gram ranking, and candidate ambiguity.
