#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a balanced, multi-focus sample of 1,000 EMLO letters around Jesuit
correspondents, then aggregate the letters into an undirected weighted network.

The script intentionally avoids taking 1,000 records from a single ego-corpus
(e.g. Athanasius Kircher). Instead it:

1. Searches the EMLO union catalogue for letters involving several Jesuit seed nodes.
2. Identifies their first-degree correspondents.
3. Adds letters from selected bridge catalogues.
4. Expands through the most frequent first-degree correspondents.
5. Keeps relations inside the Jesuit correspondence neighbourhood.
6. Deduplicates records by EMLO work UUID.
7. Draws a balanced sample with a maximum share for any single person.
8. Exports:
   - source | target | link_of_the_letter
   - an audit table with provenance/metadata
   - an aggregated weighted edge list
   - node-level network metrics
   - optionally an interactive Sigma HTML network

Important:
The script uses the Solr endpoint employed by the EMLO web interface:
    https://emlo.bodleian.ox.ac.uk/solr/works/select

This is an application endpoint rather than a formally guaranteed public API.
If EMLO changes its frontend/index structure, field names or endpoint paths may
need to be updated.

Recommended environment:
    Python 3.10+
    pip install pandas requests tqdm openpyxl networkx ipysigma

Example:
    python emlo_jesuit_network_1000.py

With options:
    python emlo_jesuit_network_1000.py \
        --sample-size 1000 \
        --max-person-share 0.20 \
        --top-expansion-people 40 \
        --output-dir data \
        --write-html
"""

from __future__ import annotations

import argparse
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

import networkx as nx
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tqdm.auto import tqdm
from urllib3.util.retry import Retry


# =============================================================================
# EMLO CONFIGURATION
# =============================================================================

BASE_URL = "https://emlo.bodleian.ox.ac.uk"
SOLR_URL = f"{BASE_URL}/solr/works/select"

DEFAULT_PAGE_SIZE = 500
DEFAULT_REQUEST_DELAY = 0.20

# Fields used by the current EMLO frontend/index.
FIELDS = [
    "uuid",
    "author_sort",
    "recipient_sort",
    "started_date_sort",
    "cito_Catalog",
]


# =============================================================================
# JESUIT SEED NODES
# =============================================================================
#
# "query" is deliberately short because it is used for broad Solr retrieval.
# "pattern" is then used locally to reject false positives.
#
# Extend this dictionary if you want a broader Jesuit seed set.
# =============================================================================

JESUIT_SEEDS = {
    "Athanasius Kircher": {
        "query": "Kircher",
        "pattern": r"\bKircher,\s*Athanasius\b",
    },
    "Caspar Schott": {
        "query": "Schott",
        "pattern": r"\bSchott,\s*(Caspar|Kaspar)\b",
    },
    "Giovanni Battista Riccioli": {
        "query": "Riccioli",
        "pattern": r"\bRiccioli,\s*Giovanni Battista\b",
    },
    "Valentin Stansel": {
        "query": "Stansel",
        "pattern": r"\bStansel,\s*Valentin(e)?\b",
    },
    "Johann Grueber": {
        "query": "Grueber",
        "pattern": r"\bGrueber,\s*Johann\b",
    },
    "Christoph Scheiner": {
        "query": "Scheiner",
        "pattern": r"\bScheiner,\s*Christoph\b",
    },
}


# =============================================================================
# BRIDGE CATALOGUES
# =============================================================================
#
# These catalogues are used to add connections beyond one Jesuit ego-network.
# A missing/renamed catalogue is skipped with a warning.
#
# You can freely add/remove catalogue names after inspecting EMLO.
# =============================================================================

BRIDGE_CATALOGUES = [
    "Kircher, Athanasius",
    "Schott, Caspar",
    "Huygens, Christiaan",
    "Mersenne, Marin",
    "Oldenburg, Henry",
    "Boyle, Robert",
    "Descartes, René",
    "Peiresc, Nicolas-Claude Fabri de",
]


# =============================================================================
# HTTP
# =============================================================================

def build_session() -> requests.Session:
    """Create an HTTP session with retries and an identifiable User-Agent."""

    session = requests.Session()

    retry = Retry(
        total=5,
        connect=5,
        read=5,
        status=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )

    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    session.headers.update(
        {
            "User-Agent": (
                "Jesuit-Network-Research/1.0 "
                "(academic metadata analysis; EMLO)"
            )
        }
    )

    return session


SESSION = build_session()


# =============================================================================
# SOLR HELPERS
# =============================================================================

def fetch_solr(
    filters: Optional[Iterable[str]] = None,
    *,
    page_size: int = DEFAULT_PAGE_SIZE,
    request_delay: float = DEFAULT_REQUEST_DELAY,
    max_records: Optional[int] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Retrieve EMLO work records from the Solr index.

    Parameters
    ----------
    filters:
        Iterable of Solr fq expressions.
    page_size:
        Number of documents per request.
    request_delay:
        Delay between paginated requests.
    max_records:
        Optional maximum number of records to retrieve.
    verbose:
        Print progress for this query.

    Returns
    -------
    pandas.DataFrame
    """

    filters = list(filters or [])
    all_docs: list[dict] = []
    start = 0

    while True:
        rows = page_size

        if max_records is not None:
            remaining = max_records - len(all_docs)
            if remaining <= 0:
                break
            rows = min(rows, remaining)

        params: list[tuple[str, object]] = [
            ("q", "*:*"),
            ("wt", "json"),
            ("rows", rows),
            ("start", start),
            ("sort", "uuid asc"),
            ("fl", ",".join(FIELDS)),
        ]

        for fq in filters:
            params.append(("fq", fq))

        response = SESSION.get(
            SOLR_URL,
            params=params,
            timeout=90,
        )

        if response.status_code >= 400:
            raise RuntimeError(
                f"EMLO/Solr returned HTTP {response.status_code}.\n"
                f"URL: {response.url}\n"
                f"Response: {response.text[:500]}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(
                "EMLO/Solr did not return JSON. "
                "The frontend endpoint may have changed."
            ) from exc

        if "response" not in payload:
            raise RuntimeError(
                "Unexpected Solr response structure. "
                "The EMLO frontend/index may have changed."
            )

        docs = payload["response"].get("docs", [])
        num_found = int(payload["response"].get("numFound", len(docs)))

        if not docs:
            break

        all_docs.extend(docs)

        if verbose:
            print(f"  retrieved {len(all_docs):,} / {num_found:,}")

        start += len(docs)

        if start >= num_found:
            break

        if max_records is not None and len(all_docs) >= max_records:
            break

        time.sleep(request_delay)

    return pd.DataFrame(all_docs)


def solr_escape_phrase(value: str) -> str:
    """Escape characters required inside a quoted Solr phrase."""
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


# =============================================================================
# DATA NORMALISATION
# =============================================================================

def scalar_value(value):
    """
    EMLO/Solr may return a scalar or a one-element list.
    For person/date fields we use the first value.
    """
    if isinstance(value, list):
        return value[0] if value else None
    return value


def join_values(value):
    """Keep all catalogue values in a readable audit string."""
    if isinstance(value, list):
        return "; ".join(str(x) for x in value)
    return value


def prepare_letters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw EMLO/Solr records into a stable letter-level table.

    One row = one EMLO work/letter.
    """

    output_columns = [
        "uuid",
        "source",
        "target",
        "date",
        "catalogue",
        "link_of_the_letter",
    ]

    if df.empty:
        return pd.DataFrame(columns=output_columns)

    df = df.copy()

    for col in FIELDS:
        if col not in df.columns:
            df[col] = None

    df["source"] = df["author_sort"].apply(scalar_value)
    df["target"] = df["recipient_sort"].apply(scalar_value)
    df["date"] = df["started_date_sort"].apply(scalar_value)
    df["catalogue"] = df["cito_Catalog"].apply(join_values)

    df["link_of_the_letter"] = (
        BASE_URL + "/profile/work/" + df["uuid"].astype(str)
    )

    # A correspondence edge requires both endpoints.
    df = df.dropna(subset=["uuid", "source", "target"])

    df["source"] = df["source"].astype(str).str.strip()
    df["target"] = df["target"].astype(str).str.strip()

    df = df[(df["source"] != "") & (df["target"] != "")]
    df = df[df["source"] != df["target"]]

    return df[output_columns].reset_index(drop=True)


# =============================================================================
# JESUIT MATCHING
# =============================================================================

def match_jesuit(name: object) -> Optional[str]:
    """
    Return the canonical seed label if a person name matches a Jesuit seed.
    Otherwise return None.
    """

    if pd.isna(name):
        return None

    text = str(name)

    for label, cfg in JESUIT_SEEDS.items():
        if re.search(cfg["pattern"], text, flags=re.IGNORECASE):
            return label

    return None


# =============================================================================
# STAGE 1: DIRECT JESUIT LETTERS
# =============================================================================

def collect_seed_letters(
    *,
    page_size: int,
    request_delay: float,
) -> pd.DataFrame:
    """
    Search the entire EMLO union catalogue for letters involving seed Jesuits.
    """

    frames: list[pd.DataFrame] = []

    print("\n[1/7] Searching EMLO for Jesuit seed nodes...")

    for jesuit, cfg in JESUIT_SEEDS.items():
        print(f"\n  Seed: {jesuit}")

        term = cfg["query"]

        # Broad remote query...
        fq = f"(author_sort:*{term}* OR recipient_sort:*{term}*)"

        try:
            raw = fetch_solr(
                [fq],
                page_size=page_size,
                request_delay=request_delay,
            )
        except Exception as exc:
            print(f"    WARNING: query failed: {exc}")
            continue

        letters = prepare_letters(raw)

        if letters.empty:
            print("    found 0 usable letters")
            continue

        # ...followed by stricter local validation.
        mask = (
            letters["source"]
            .str.contains(
                cfg["pattern"],
                regex=True,
                case=False,
                na=False,
            )
            |
            letters["target"]
            .str.contains(
                cfg["pattern"],
                regex=True,
                case=False,
                na=False,
            )
        )

        letters = letters.loc[mask].copy()
        letters["seed_query"] = jesuit
        letters["source_stage"] = "seed"

        print(f"    found {len(letters):,} usable letters")
        frames.append(letters)

    if not frames:
        raise RuntimeError(
            "No Jesuit seed letters were retrieved. "
            "Check the EMLO endpoint, index field names, and seed patterns."
        )

    result = pd.concat(frames, ignore_index=True)

    # One work may have been found through several seed queries.
    result = (
        result
        .sort_values("uuid")
        .drop_duplicates("uuid", keep="first")
        .reset_index(drop=True)
    )

    result["source_jesuit"] = result["source"].apply(match_jesuit)
    result["target_jesuit"] = result["target"].apply(match_jesuit)
    result["direct_jesuit"] = (
        result["source_jesuit"].notna()
        | result["target_jesuit"].notna()
    )

    print(f"\n  Unique direct Jesuit letters: {len(result):,}")
    return result


# =============================================================================
# STAGE 2: FIRST-DEGREE CORRESPONDENTS
# =============================================================================

def get_first_degree_correspondents(
    seed_letters: pd.DataFrame,
) -> tuple[set[str], set[str]]:
    """
    Return:
      1. non-seed correspondents directly connected to a Jesuit seed;
      2. exact EMLO name variants of the Jesuit seeds found in the data.
    """

    print("\n[2/7] Identifying first-degree correspondents...")

    first_degree: set[str] = set()
    jesuit_names_in_data: set[str] = set()

    for _, row in seed_letters.iterrows():
        source = row["source"]
        target = row["target"]

        source_jesuit = match_jesuit(source)
        target_jesuit = match_jesuit(target)

        if source_jesuit:
            jesuit_names_in_data.add(source)
        else:
            first_degree.add(source)

        if target_jesuit:
            jesuit_names_in_data.add(target)
        else:
            first_degree.add(target)

    print(f"  First-degree correspondents: {len(first_degree):,}")
    print(f"  Jesuit name variants in EMLO: {len(jesuit_names_in_data):,}")

    return first_degree, jesuit_names_in_data


# =============================================================================
# STAGE 3: BRIDGE CATALOGUES
# =============================================================================

def collect_bridge_catalogues(
    *,
    page_size: int,
    request_delay: float,
) -> pd.DataFrame:
    """
    Retrieve selected correspondence catalogues that can supply ties among
    Jesuit correspondents and the wider Republic of Letters.
    """

    print("\n[3/7] Retrieving bridge catalogues...")

    frames: list[pd.DataFrame] = []

    for catalogue in tqdm(BRIDGE_CATALOGUES, desc="  catalogues"):
        safe_catalogue = solr_escape_phrase(catalogue)
        fq = f'cito_Catalog:"{safe_catalogue}"'

        try:
            raw = fetch_solr(
                [fq],
                page_size=page_size,
                request_delay=request_delay,
            )
        except Exception as exc:
            print(f"\n  WARNING: {catalogue}: {exc}")
            continue

        letters = prepare_letters(raw)

        if letters.empty:
            print(f"\n  WARNING: no records found for catalogue: {catalogue}")
            continue

        letters["queried_catalogue"] = catalogue
        letters["source_stage"] = "bridge_catalogue"
        frames.append(letters)

    if not frames:
        print(
            "  WARNING: no bridge catalogue records retrieved. "
            "The script will continue with seed/expansion data."
        )
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    result = result.drop_duplicates("uuid").reset_index(drop=True)

    print(f"  Unique bridge-catalogue letters: {len(result):,}")
    return result


# =============================================================================
# STAGE 4: EXPANSION THROUGH MAJOR CORRESPONDENTS
# =============================================================================

def rank_correspondents(seed_letters: pd.DataFrame) -> list[str]:
    """
    Rank non-seed correspondents by number of direct letters with seed Jesuits.
    """

    counts: Counter[str] = Counter()

    for _, row in seed_letters.iterrows():
        source = row["source"]
        target = row["target"]

        source_is_jesuit = match_jesuit(source) is not None
        target_is_jesuit = match_jesuit(target) is not None

        if source_is_jesuit and not target_is_jesuit:
            counts[target] += 1

        if target_is_jesuit and not source_is_jesuit:
            counts[source] += 1

    return [person for person, _ in counts.most_common()]


def collect_expansion_letters(
    people: list[str],
    *,
    max_records_per_person: int,
    page_size: int,
    request_delay: float,
) -> pd.DataFrame:
    """
    Retrieve letters involving the most frequent first-degree correspondents.
    """

    print(
        f"\n[4/7] Expanding through {len(people):,} "
        "major first-degree correspondents..."
    )

    frames: list[pd.DataFrame] = []

    for person in tqdm(people, desc="  correspondents"):
        safe_person = solr_escape_phrase(person)

        fq = (
            f'(author_sort:"{safe_person}" '
            f'OR recipient_sort:"{safe_person}")'
        )

        try:
            raw = fetch_solr(
                [fq],
                page_size=page_size,
                request_delay=request_delay,
                max_records=max_records_per_person,
            )
        except Exception as exc:
            print(f"\n  WARNING: {person}: {exc}")
            continue

        letters = prepare_letters(raw)

        if letters.empty:
            continue

        letters["expansion_person"] = person
        letters["source_stage"] = "correspondent_expansion"
        frames.append(letters)

    if not frames:
        print("  WARNING: no expansion records retrieved.")
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    result = result.drop_duplicates("uuid").reset_index(drop=True)

    print(f"  Unique expansion letters retrieved: {len(result):,}")
    return result


# =============================================================================
# STAGE 5: BUILD CANDIDATE NETWORK
# =============================================================================

def build_candidate_pool(
    seed_letters: pd.DataFrame,
    first_degree: set[str],
    jesuit_names_in_data: set[str],
    bridge_letters: pd.DataFrame,
    expansion_letters: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a candidate letter pool.

    Seed letters are always retained.

    Bridge/expansion letters are retained when BOTH endpoints are already in the
    first-degree Jesuit neighbourhood. This is deliberate: the expansion is used
    to discover ties among known Jesuit correspondents, rather than recursively
    exploding into the entire Republic of Letters.
    """

    print("\n[5/7] Building the multi-focus candidate pool...")

    network_people = first_degree | jesuit_names_in_data

    frames: list[pd.DataFrame] = [seed_letters.copy()]

    if not bridge_letters.empty:
        bridge_network = bridge_letters[
            bridge_letters["source"].isin(network_people)
            & bridge_letters["target"].isin(network_people)
        ].copy()

        print(
            "  Bridge letters inside Jesuit neighbourhood: "
            f"{len(bridge_network):,}"
        )

        frames.append(bridge_network)

    if not expansion_letters.empty:
        expansion_network = expansion_letters[
            expansion_letters["source"].isin(network_people)
            & expansion_letters["target"].isin(network_people)
        ].copy()

        print(
            "  Expansion letters inside Jesuit neighbourhood: "
            f"{len(expansion_network):,}"
        )

        frames.append(expansion_network)

    candidates = pd.concat(frames, ignore_index=True)

    # If a letter occurs in several catalogues/stages, retain one row by UUID.
    candidates = (
        candidates
        .sort_values(["uuid", "source_stage"])
        .drop_duplicates("uuid", keep="first")
        .reset_index(drop=True)
    )

    candidates["source_jesuit"] = candidates["source"].apply(match_jesuit)
    candidates["target_jesuit"] = candidates["target"].apply(match_jesuit)
    candidates["direct_jesuit"] = (
        candidates["source_jesuit"].notna()
        | candidates["target_jesuit"].notna()
    )

    print(f"  Candidate letters: {len(candidates):,}")
    print(
        "  Direct Jesuit letters in candidate pool: "
        f"{int(candidates['direct_jesuit'].sum()):,}"
    )

    return candidates


# =============================================================================
# STAGE 6: BALANCED SAMPLE
# =============================================================================

def balanced_sample(
    df: pd.DataFrame,
    *,
    n: int,
    max_person_share: float,
    min_direct_jesuit_letters: int,
    random_state: int,
) -> pd.DataFrame:
    """
    Draw a reproducible sample while limiting domination by one person.

    The cap counts how many selected letters are incident to a person, whether
    that person is sender or recipient.

    Example:
        n = 1000
        max_person_share = 0.20
        => no person may occur in more than 200 selected letters.

    The sampler first tries to ensure a minimum number of direct Jesuit letters,
    then fills remaining slots from the entire candidate pool.

    Note:
    A strict person cap may make n=1000 mathematically impossible for a sparse
    candidate pool. In that case the function fails explicitly rather than
    silently relaxing the methodological constraint.
    """

    if not 0 < max_person_share <= 1:
        raise ValueError("--max-person-share must be in (0, 1].")

    if len(df) < n:
        raise ValueError(
            f"Only {len(df):,} unique candidate letters are available, "
            f"but {n:,} were requested.\n"
            "Increase --top-expansion-people, "
            "--max-records-per-expansion-person, or expand the seed/catalogue set."
        )

    person_cap = max(1, int(n * max_person_share))
    rng = random.Random(random_state)

    selected_indices: list[int] = []
    selected_ids: set[str] = set()
    person_counter: Counter[str] = Counter()

    def can_add(row: pd.Series) -> bool:
        return (
            person_counter[row["source"]] < person_cap
            and person_counter[row["target"]] < person_cap
        )

    def add_row(idx: int, row: pd.Series) -> None:
        selected_indices.append(idx)
        selected_ids.add(row["uuid"])
        person_counter[row["source"]] += 1
        person_counter[row["target"]] += 1

    # A. Direct Jesuit correspondence first.
    direct_indices = list(df.index[df["direct_jesuit"]])
    rng.shuffle(direct_indices)

    direct_added = 0

    for idx in direct_indices:
        if direct_added >= min_direct_jesuit_letters:
            break

        row = df.loc[idx]

        if row["uuid"] not in selected_ids and can_add(row):
            add_row(idx, row)
            direct_added += 1

    if direct_added < min_direct_jesuit_letters:
        print(
            "  WARNING: requested minimum of "
            f"{min_direct_jesuit_letters:,} direct Jesuit letters, "
            f"but only {direct_added:,} could be selected under the person cap."
        )

    # B. Fill from the full candidate pool.
    remaining = [
        idx
        for idx in df.index
        if df.loc[idx, "uuid"] not in selected_ids
    ]
    rng.shuffle(remaining)

    for idx in remaining:
        if len(selected_indices) >= n:
            break

        row = df.loc[idx]

        if can_add(row):
            add_row(idx, row)

    if len(selected_indices) < n:
        top_counts = person_counter.most_common(10)

        raise ValueError(
            "\nBalanced sampling could not reach the requested sample size.\n"
            f"Requested: {n:,}\n"
            f"Selected: {len(selected_indices):,}\n"
            f"Per-person cap: {person_cap:,} letters "
            f"({max_person_share:.1%} of the sample)\n"
            f"Most frequent selected people: {top_counts}\n\n"
            "Recommended fix: broaden the network by increasing "
            "--top-expansion-people or adding more Jesuit seeds/bridge catalogues. "
            "Relax --max-person-share only if you explicitly accept a more "
            "ego-centred teaching sample."
        )

    result = (
        df.loc[selected_indices]
        .copy()
        .reset_index(drop=True)
    )

    return result


# =============================================================================
# NETWORK CONSTRUCTION
# =============================================================================

def aggregate_undirected_edges(
    letters: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate letter-level rows into undirected weighted person-person edges.

    weight = number of letters between the two people in the selected sample.
    """

    pairs = np.sort(
        letters[["source", "target"]].astype(str).values,
        axis=1,
    )

    temp = letters.copy()
    temp["person"] = pairs[:, 0]
    temp["response_person"] = pairs[:, 1]

    edges = (
        temp
        .groupby(["person", "response_person"], as_index=False)
        .agg(
            weight=("uuid", "size"),
            example_letter=("link_of_the_letter", "first"),
        )
        .sort_values(
            ["weight", "person", "response_person"],
            ascending=[False, True, True],
        )
        .reset_index(drop=True)
    )

    return edges


def build_graph(edges: pd.DataFrame) -> nx.Graph:
    """Build the undirected weighted NetworkX graph."""

    graph = nx.Graph()

    for _, row in tqdm(
        edges.iterrows(),
        total=len(edges),
        desc="  network edges",
    ):
        person = row["person"]
        response_person = row["response_person"]
        weight = int(row["weight"])

        graph.add_node(person, type="person")
        graph.add_node(response_person, type="person")

        graph.add_edge(
            person,
            response_person,
            weight=weight,
            example_letter=row["example_letter"],
        )

    # Raw degree: number of distinct neighbours in the sampled graph.
    degree_dict = dict(graph.degree())
    nx.set_node_attributes(graph, degree_dict, name="degree")

    # Weighted degree: number of sampled letters incident to the node.
    weighted_degree_dict = dict(graph.degree(weight="weight"))
    nx.set_node_attributes(
        graph,
        weighted_degree_dict,
        name="weighted_degree",
    )

    # Standard centralities.
    nx.set_node_attributes(
        graph,
        nx.degree_centrality(graph),
        "degree_centrality",
    )

    nx.set_node_attributes(
        graph,
        nx.betweenness_centrality(graph),
        "betweenness",
    )

    nx.set_node_attributes(
        graph,
        nx.closeness_centrality(graph),
        "closeness",
    )

    try:
        eigenvector = nx.eigenvector_centrality(
            graph,
            max_iter=5000,
            tol=1.0e-8,
        )
    except nx.PowerIterationFailedConvergence:
        print(
            "  WARNING: eigenvector centrality did not converge; "
            "eigenvector values will be NaN."
        )
        eigenvector = {node: np.nan for node in graph.nodes}

    nx.set_node_attributes(
        graph,
        eigenvector,
        "eigenvector",
    )

    return graph


def node_metrics_table(graph: nx.Graph) -> pd.DataFrame:
    """Export node-level metrics to a flat table."""

    rows = []

    for node, attrs in graph.nodes(data=True):
        rows.append(
            {
                "person": node,
                "degree": attrs.get("degree"),
                "weighted_degree": attrs.get("weighted_degree"),
                "degree_centrality": attrs.get("degree_centrality"),
                "betweenness": attrs.get("betweenness"),
                "closeness": attrs.get("closeness"),
                "eigenvector": attrs.get("eigenvector"),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["degree", "weighted_degree"],
            ascending=False,
        )
        .reset_index(drop=True)
    )


# =============================================================================
# QUALITY CONTROL / REPORT
# =============================================================================

def print_sample_report(sample: pd.DataFrame) -> None:
    """Print basic diagnostics before network interpretation."""

    print("\n[6/7] Sample diagnostics")

    people = pd.concat(
        [sample["source"], sample["target"]],
        ignore_index=True,
    )

    incident_counts = people.value_counts()
    incident_share = incident_counts / len(sample)

    unique_people = len(set(sample["source"]) | set(sample["target"]))

    print(f"  Letters: {len(sample):,}")
    print(f"  Unique people: {unique_people:,}")
    print(
        "  Direct Jesuit letters: "
        f"{int(sample['direct_jesuit'].sum()):,}"
    )

    if not incident_share.empty:
        print(
            "  Maximum single-person share: "
            f"{incident_share.iloc[0]:.2%} "
            f"({incident_share.index[0]})"
        )

    print("\n  Top 15 people by number of incident sampled letters:")
    top_people = pd.DataFrame(
        {
            "person": incident_counts.head(15).index,
            "letters": incident_counts.head(15).values,
            "share": incident_share.head(15).values,
        }
    )
    print(top_people.to_string(index=False))

    if "catalogue" in sample.columns:
        catalogue_counts = (
            sample["catalogue"]
            .fillna("[no catalogue value]")
            .value_counts()
            .head(15)
        )

        print("\n  Top catalogue values in the sample:")
        print(catalogue_counts.to_string())


# =============================================================================
# EXPORT
# =============================================================================

def export_outputs(
    sample: pd.DataFrame,
    edges: pd.DataFrame,
    graph: nx.Graph,
    output_dir: Path,
    *,
    sample_size: int,
    write_html: bool,
) -> None:
    """Write teaching table, audit table, edges, metrics and graph files."""

    output_dir.mkdir(parents=True, exist_ok=True)

    stem = f"emlo_jesuit_network_{sample_size}"

    # Exact teaching structure requested by the user.
    final_table = sample[
        ["source", "target", "link_of_the_letter"]
    ].copy()

    final_csv = output_dir / f"{stem}.csv"
    final_xlsx = output_dir / f"{stem}.xlsx"
    audit_csv = output_dir / f"{stem}_audit.csv"
    edges_csv = output_dir / f"{stem}_edges.csv"
    metrics_csv = output_dir / f"{stem}_metrics.csv"
    graphml_path = output_dir / f"{stem}.graphml"

    final_table.to_csv(
        final_csv,
        index=False,
        encoding="utf-8-sig",
    )

    final_table.to_excel(
        final_xlsx,
        index=False,
    )

    sample.to_csv(
        audit_csv,
        index=False,
        encoding="utf-8-sig",
    )

    edges.to_csv(
        edges_csv,
        index=False,
        encoding="utf-8-sig",
    )

    metrics = node_metrics_table(graph)

    metrics.to_csv(
        metrics_csv,
        index=False,
        encoding="utf-8-sig",
    )

    nx.write_graphml(graph, graphml_path)

    print("\n[7/7] Files written:")
    for path in [
        final_csv,
        final_xlsx,
        audit_csv,
        edges_csv,
        metrics_csv,
        graphml_path,
    ]:
        print(f"  {path}")

    if write_html:
        try:
            from ipysigma import Sigma
        except ImportError:
            print(
                "\n  WARNING: --write-html requested, but ipysigma "
                "is not installed. Run: pip install ipysigma"
            )
            return

        html_path = output_dir / f"{stem}.html"

        try:
            Sigma.write_html(
                graph,
                str(html_path),
                fullscreen=True,
                node_metrics=["louvain"],
                node_color="louvain",
                node_size="degree",
                node_size_range=(3, 25),
                max_categorical_colors=30,
                default_edge_type="curve",
                default_node_label_size=14,
                node_border_color_from="node",
            )
            print(f"  {html_path}")
        except Exception as exc:
            print(
                "\n  WARNING: Sigma HTML export failed. "
                f"The other outputs were written successfully.\n  {exc}"
            )


# =============================================================================
# MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a balanced multi-focus EMLO correspondence sample "
            "around Jesuit seed nodes."
        )
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Number of unique letters in the final sample (default: 1000).",
    )

    parser.add_argument(
        "--max-person-share",
        type=float,
        default=0.20,
        help=(
            "Maximum fraction of sampled letters incident to any one person "
            "(default: 0.20)."
        ),
    )

    parser.add_argument(
        "--min-direct-jesuit-letters",
        type=int,
        default=250,
        help=(
            "Target minimum number of letters directly involving seed Jesuits "
            "(default: 250). Subject to the per-person cap."
        ),
    )

    parser.add_argument(
        "--top-expansion-people",
        type=int,
        default=40,
        help=(
            "Number of most frequent first-degree correspondents used for "
            "additional EMLO expansion queries (default: 40)."
        ),
    )

    parser.add_argument(
        "--max-records-per-expansion-person",
        type=int,
        default=1500,
        help=(
            "Safety limit for records retrieved for each expansion person "
            "(default: 1500)."
        ),
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=2026,
        help="Random seed for reproducible sampling (default: 2026).",
    )

    parser.add_argument(
        "--page-size",
        type=int,
        default=DEFAULT_PAGE_SIZE,
        help=f"Solr page size (default: {DEFAULT_PAGE_SIZE}).",
    )

    parser.add_argument(
        "--request-delay",
        type=float,
        default=DEFAULT_REQUEST_DELAY,
        help=(
            "Delay in seconds between paginated EMLO requests "
            f"(default: {DEFAULT_REQUEST_DELAY})."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory for outputs (default: ./data).",
    )

    parser.add_argument(
        "--write-html",
        action="store_true",
        help="Also export an interactive Sigma HTML network.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.sample_size <= 0:
        raise ValueError("--sample-size must be > 0.")

    if args.min_direct_jesuit_letters < 0:
        raise ValueError("--min-direct-jesuit-letters must be >= 0.")

    if args.top_expansion_people < 0:
        raise ValueError("--top-expansion-people must be >= 0.")

    print("=" * 78)
    print("EMLO JESUIT CORRESPONDENCE NETWORK")
    print("=" * 78)
    print(f"EMLO endpoint: {SOLR_URL}")
    print(f"Requested sample: {args.sample_size:,} unique letters")
    print(f"Maximum person share: {args.max_person_share:.1%}")
    print(f"Random state: {args.random_state}")

    # 1. Direct letters involving Jesuit seed nodes.
    seed_letters = collect_seed_letters(
        page_size=args.page_size,
        request_delay=args.request_delay,
    )

    # 2. First-degree neighbourhood.
    first_degree, jesuit_names_in_data = get_first_degree_correspondents(
        seed_letters
    )

    # 3. Selected independent/bridge catalogues.
    bridge_letters = collect_bridge_catalogues(
        page_size=args.page_size,
        request_delay=args.request_delay,
    )

    # 4. Expansion through major Jesuit correspondents.
    ranked_correspondents = rank_correspondents(seed_letters)
    top_people = ranked_correspondents[: args.top_expansion_people]

    expansion_letters = collect_expansion_letters(
        top_people,
        max_records_per_person=args.max_records_per_expansion_person,
        page_size=args.page_size,
        request_delay=args.request_delay,
    )

    # 5. Candidate pool.
    candidates = build_candidate_pool(
        seed_letters=seed_letters,
        first_degree=first_degree,
        jesuit_names_in_data=jesuit_names_in_data,
        bridge_letters=bridge_letters,
        expansion_letters=expansion_letters,
    )

    # 6. Balanced sampling.
    print("\n  Drawing balanced sample...")

    sample = balanced_sample(
        candidates,
        n=args.sample_size,
        max_person_share=args.max_person_share,
        min_direct_jesuit_letters=args.min_direct_jesuit_letters,
        random_state=args.random_state,
    )

    print_sample_report(sample)

    # 7. Aggregate letter-level records and calculate network metrics.
    print("\n  Aggregating letters into undirected weighted edges...")

    edges = aggregate_undirected_edges(sample)
    graph = build_graph(edges)

    print(f"  Graph nodes: {graph.number_of_nodes():,}")
    print(f"  Graph edges: {graph.number_of_edges():,}")
    print(
        "  Connected components: "
        f"{nx.number_connected_components(graph):,}"
    )

    export_outputs(
        sample=sample,
        edges=edges,
        graph=graph,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        write_html=args.write_html,
    )

    print("\nDone.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
