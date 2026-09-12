from __future__ import annotations

import re
import unicodedata

# Deliberately small reference set. Extend with your audited domain rules.
ABBREVIATIONS = {
    "ins": "insurance",
    "insur": "insurance",
    "co": "company",
    "corp": "corporation",
    "grp": "group",
    "intl": "international",
    "natl": "national",
    "mut": "mutual",
}

# Conservative removal: only legal-form noise that is rarely identity-bearing.
CONSERVATIVE_STOPWORDS = {
    "inc",
    "incorporated",
    "llc",
    "ltd",
    "limited",
}

# Aggressive removal: useful for candidate generation, but potentially lossy.
AGGRESSIVE_STOPWORDS = CONSERVATIVE_STOPWORDS | {
    "insurance",
    "company",
    "corporation",
    "group",
}


def _base_normalize(text: str) -> str:
    """Loss-minimizing normalization shared by reference and user strings.

    Steps:
      1. Unicode NFKD normalization and ASCII folding.
      2. Lowercase.
      3. Replace '&' with 'and'.
      4. Remove punctuation/non-alphanumeric characters.
      5. Normalize whitespace.
      6. Expand audited whole-token abbreviations.

    The exact same base transform should be applied to AM Best reference names
    and user input. Do not use fuzzy matching on raw strings while comparing
    against differently normalized references.
    """
    if text is None:
        return ""

    text = unicodedata.normalize("NFKD", str(text))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = [ABBREVIATIONS.get(tok, tok) for tok in text.split()]
    return " ".join(tokens)


def _drop_tokens(text: str, stopwords: set[str]) -> str:
    return " ".join(tok for tok in text.split() if tok not in stopwords)


def normalize_conservative(text: str) -> str:
    """Primary representation for both AM Best names and user inputs.

    This should preserve nearly all identity-bearing words. Use it for exact
    deterministic matching and as the primary representation for fuzzy
    features.
    """
    return _drop_tokens(_base_normalize(text), CONSERVATIVE_STOPWORDS)


def normalize_aggressive(text: str) -> str:
    """Lossy secondary representation for both references and user inputs.

    This removes generic insurance/legal tokens. It should be treated as
    additional evidence, not as the sole source of an automatic match unless
    the normalized value is known to identify exactly one group.
    """
    return _drop_tokens(_base_normalize(text), AGGRESSIVE_STOPWORDS)
