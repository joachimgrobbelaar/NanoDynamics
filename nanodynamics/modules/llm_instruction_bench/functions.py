"""
nanodynamics/modules/llm_instruction_bench/functions.py

Pillar 7: LLM Instruction Tuning, Astrodynamics QA & Reasoning.

Provides functions for loading and transforming astrodynamics instruction-
tuning datasets into formats suitable for supervised fine-tuning (ChatML,
Alpaca), filtering by mathematical complexity, and building word-frequency
distributions from mission-operations pretraining corpora.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any, Dict, List, Literal, Optional, Tuple

from nanodynamics.data_ingestion.loaders import get_dataset

logger = logging.getLogger("nanodynamics.modules.llm_instruction_bench")

__all__ = [
    "load_orbital_instruction_datasets",
    "build_astrodynamics_sft_dataset",
    "filter_by_math_complexity",
    "load_momo_pretraining_corpus",
    "tokenize_mission_operations_corpus",
]

# ============================================================================
# Constants
# ============================================================================

_INSTRUCTION_DATASET_KEYS: List[str] = [
    "orbital_mechanics_1",
    "orbital_mechanics_instruct_32",
    "ssao_space_instruct",
]

# Known column-name variants across the three instruction datasets.
# Each tuple lists aliases in priority order; the first match wins.
_INSTRUCTION_ALIASES: List[Tuple[str, ...]] = (
    ("instruction", "question", "prompt", "input", "query", "user"),
)
_CONTEXT_ALIASES: List[Tuple[str, ...]] = (
    ("context", "input", "system", "system_prompt"),
)
_RESPONSE_ALIASES: List[Tuple[str, ...]] = (
    ("response", "output", "answer", "completion", "assistant", "text"),
)
_SYSTEM_ALIASES: List[Tuple[str, ...]] = (
    ("system", "system_prompt", "system_message"),
)

# LaTeX markers used for mathematical-complexity scoring.
_LATEX_PATTERNS: List[str] = [
    r"\$",        # inline or display math delimiter
    r"\\\\",      # literal backslash (LaTeX command prefix)
    r"\\frac",
    r"\\int",
    r"\\sum",
    r"\\sqrt",
]
_LATEX_RE = re.compile("|".join(_LATEX_PATTERNS))

# Default system prompt injected for ChatML records when none is present.
_DEFAULT_SYSTEM_PROMPT = (
    "You are an expert astrodynamics assistant specialising in orbital "
    "mechanics, celestial dynamics, and space mission design."
)


# ============================================================================
# Helpers
# ============================================================================

def _resolve_field(record: Dict[str, Any], aliases: Tuple[str, ...]) -> Optional[str]:
    """Return the value for the first alias that exists and is non-empty."""
    for alias in aliases:
        value = record.get(alias)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _count_latex_chars(text: str) -> int:
    """Count the total number of LaTeX-marker character occurrences in *text*.

    Each regex match contributes the length of the matched substring so that
    multi-character tokens (e.g. ``\\frac``) are weighted by their actual
    character count rather than counted as 1.
    """
    return sum(len(m.group()) for m in _LATEX_RE.finditer(text))


def _dataset_to_records(dataset_key: str) -> List[Dict[str, Any]]:
    """Load a single registered dataset and materialise it as a list of dicts.

    Handles both ``Dataset`` (Arrow-backed) and ``IterableDataset`` objects
    returned by :func:`get_dataset`.
    """
    logger.info("Loading dataset '%s' …", dataset_key)
    ds = get_dataset(dataset_key)

    # ``Dataset`` supports direct iteration; ``IterableDataset`` likewise.
    # Both yield dict-like rows, but a regular ``Dataset`` can be batch-
    # converted more efficiently.
    try:
        # Fast path for in-memory Arrow datasets.
        records: List[Dict[str, Any]] = [dict(row) for row in ds]
    except Exception:
        # Fallback for exotic dataset wrappers.
        records = list(ds)

    logger.info("Loaded %d records from '%s'.", len(records), dataset_key)
    return records


# ============================================================================
# 1. load_orbital_instruction_datasets
# ============================================================================

def load_orbital_instruction_datasets() -> List[Dict[str, Any]]:
    """Load and concatenate the three canonical orbital-mechanics instruction
    datasets.

    Datasets combined:
        * ``orbital_mechanics_1`` – vanta-research/orbital-mechanics-1
        * ``orbital_mechanics_instruct_32`` – abdohisham12/orbital-mechanics-instruct-32
        * ``ssao_space_instruct`` – fabsssss/ssao-space-instruct

    Returns
    -------
    List[Dict[str, Any]]
        A flat list of raw record dictionaries, preserving the original column
        names of each source dataset.
    """
    combined: List[Dict[str, Any]] = []
    for key in _INSTRUCTION_DATASET_KEYS:
        records = _dataset_to_records(key)
        # Tag every record with its provenance so downstream consumers can
        # distinguish origin when column schemas differ.
        for rec in records:
            rec.setdefault("_source_dataset", key)
        combined.extend(records)

    logger.info(
        "Combined orbital instruction corpus: %d records from %d datasets.",
        len(combined),
        len(_INSTRUCTION_DATASET_KEYS),
    )
    return combined


# ============================================================================
# 2. build_astrodynamics_sft_dataset
# ============================================================================

def build_astrodynamics_sft_dataset(
    ds_records: List[Dict[str, Any]],
    format: Literal["chatml", "alpaca"] = "chatml",
) -> List[Dict[str, Any]]:
    """Normalise instruction/response pairs into a standard SFT format.

    Parameters
    ----------
    ds_records : List[Dict[str, Any]]
        Raw records as returned by :func:`load_orbital_instruction_datasets`.
        Column names may vary across source datasets; this function resolves
        known aliases automatically.
    format : ``"chatml"`` | ``"alpaca"``
        Target serialisation layout.

        * **ChatML** – ``{"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}``
        * **Alpaca** – ``{"instruction": ..., "input": ..., "output": ...}``

    Returns
    -------
    List[Dict[str, Any]]
        Normalised records in the requested format.  Records for which no
        instruction *or* response could be resolved are dropped, and a warning
        is logged.
    """
    if format not in ("chatml", "alpaca"):
        raise ValueError(
            f"Unsupported format '{format}'. Choose 'chatml' or 'alpaca'."
        )

    normalised: List[Dict[str, Any]] = []
    skipped = 0

    for idx, rec in enumerate(ds_records):
        instruction = _resolve_field(rec, _INSTRUCTION_ALIASES[0])
        response = _resolve_field(rec, _RESPONSE_ALIASES[0])

        if not instruction or not response:
            skipped += 1
            continue

        context = _resolve_field(rec, _CONTEXT_ALIASES[0]) or ""
        system_msg = _resolve_field(rec, _SYSTEM_ALIASES[0]) or _DEFAULT_SYSTEM_PROMPT

        if format == "chatml":
            messages: List[Dict[str, str]] = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": instruction},
            ]
            # If a non-trivial context exists and differs from the
            # instruction, prepend it to the user turn.
            if context and context != instruction:
                messages[1]["content"] = f"{context}\n\n{instruction}"

            messages.append({"role": "assistant", "content": response})
            normalised.append({"messages": messages})

        else:  # alpaca
            normalised.append({
                "instruction": instruction,
                "input": context,
                "output": response,
            })

    if skipped:
        logger.warning(
            "Dropped %d / %d records lacking instruction or response fields.",
            skipped,
            len(ds_records),
        )

    logger.info(
        "Built SFT dataset: %d records in '%s' format.",
        len(normalised),
        format,
    )
    return normalised


# ============================================================================
# 3. filter_by_math_complexity
# ============================================================================

def filter_by_math_complexity(
    ds_records: List[Dict[str, Any]],
    min_latex_chars: int = 50,
) -> List[Dict[str, Any]]:
    """Filter records whose response/output contains substantive LaTeX content.

    The function inspects the *response* (or *output*) field of each record and
    counts the total characters contributed by LaTeX markers (``$``, ``\\``,
    ``\\frac``, ``\\int``, ``\\sum``, ``\\sqrt``).  Records whose cumulative
    count meets or exceeds *min_latex_chars* are retained.

    Parameters
    ----------
    ds_records : List[Dict[str, Any]]
        Records in any layout—raw, ChatML, or Alpaca.
    min_latex_chars : int
        Minimum total LaTeX-marker characters required to pass the filter.

    Returns
    -------
    List[Dict[str, Any]]
        Filtered subset of *ds_records*.
    """
    filtered: List[Dict[str, Any]] = []

    for rec in ds_records:
        # Attempt to locate the response text regardless of format.
        response_text: Optional[str] = None

        # ChatML format: extract assistant turn.
        messages = rec.get("messages")
        if isinstance(messages, list):
            for msg in messages:
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    response_text = msg.get("content", "")
                    break

        # Alpaca format or raw record.
        if response_text is None:
            response_text = _resolve_field(rec, _RESPONSE_ALIASES[0]) or ""

        latex_count = _count_latex_chars(response_text)
        if latex_count >= min_latex_chars:
            filtered.append(rec)

    logger.info(
        "LaTeX complexity filter (min=%d): kept %d / %d records.",
        min_latex_chars,
        len(filtered),
        len(ds_records),
    )
    return filtered


# ============================================================================
# 4. load_momo_pretraining_corpus / tokenize_mission_operations_corpus
# ============================================================================

def load_momo_pretraining_corpus() -> List[Dict[str, Any]]:
    """Load the MOMO (Mission Operations & Management Ontology) pretraining
    corpus.

    Returns
    -------
    List[Dict[str, Any]]
        Raw records from the ``momo_pretraining`` dataset
        (Mirali33/MOMO-pretraining-data).
    """
    return _dataset_to_records("momo_pretraining")


def tokenize_mission_operations_corpus(
    corpus_records: List[Dict[str, Any]],
    vocab_size: int = 32_000,
) -> List[Tuple[str, int]]:
    """Build a word-frequency distribution from the corpus text fields.

    This is a *lightweight stub* intended for exploratory analysis and
    vocabulary auditing.  For production tokeniser training (BPE / Unigram),
    integrate with ``sentencepiece`` or ``tokenizers`` and replace this
    function.

    Parameters
    ----------
    corpus_records : List[Dict[str, Any]]
        Records as returned by :func:`load_momo_pretraining_corpus`.  The
        function searches for a text payload in common field names
        (``text``, ``content``, ``document``, ``passage``).
    vocab_size : int
        Maximum number of unique tokens to return, ranked by descending
        frequency.

    Returns
    -------
    List[Tuple[str, int]]
        The top-*vocab_size* ``(token, count)`` pairs sorted by frequency
        (descending).
    """
    # Known field names that may contain the main text payload.
    _TEXT_FIELD_ALIASES = ("text", "content", "document", "passage", "body")

    word_counter: Counter = Counter()
    records_with_text = 0

    for rec in corpus_records:
        text: Optional[str] = None
        for alias in _TEXT_FIELD_ALIASES:
            val = rec.get(alias)
            if val and isinstance(val, str) and val.strip():
                text = val
                break

        if text is None:
            continue

        records_with_text += 1
        # Simple whitespace tokenisation; lowercase for frequency merging.
        # TODO: Replace with sentencepiece / HF tokenizers BPE trainer for
        #       production-grade sub-word vocabulary.
        tokens = re.findall(r"[A-Za-z0-9]+(?:'[a-z]+)?", text.lower())
        word_counter.update(tokens)

    logger.info(
        "Tokenised %d records (%d had extractable text). "
        "Unique tokens: %d. Returning top %d.",
        len(corpus_records),
        records_with_text,
        len(word_counter),
        min(vocab_size, len(word_counter)),
    )

    return word_counter.most_common(vocab_size)
