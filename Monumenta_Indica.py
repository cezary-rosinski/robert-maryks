from pathlib import Path
from itertools import combinations
import re

import pandas as pd
import networkx as nx
from tqdm import tqdm

try:
    from ipysigma import Sigma
except ImportError:
    Sigma = None


# ============================================================
# 1. KONFIGURACJA
# ============================================================

# Folder wejściowy jest liczony względem lokalizacji tego skryptu.
# Dzięki temu kod działa niezależnie od nazwy konta Windows.
SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
INPUT_FOLDER = SCRIPT_DIR / "data" / "Monumenta Indica"

OUTPUT_FOLDER = INPUT_FOLDER / "network_output"
RELATIONS_XLSX = OUTPUT_FOLDER / "Monumenta_Indica_relations_final.xlsx"
OCCURRENCES_XLSX = OUTPUT_FOLDER / "Monumenta_Indica_occurrences_long.xlsx"
GRAPHML_PATH = OUTPUT_FOLDER / "Monumenta_Indica.graphml"
HTML_PATH = OUTPUT_FOLDER / "Monumenta_Indica_network.html"
HTML_PAGERANK_PATH = OUTPUT_FOLDER / "Monumenta_Indica_network_pagerank.html"
CO_OCCURRENCE_XLSX = OUTPUT_FOLDER / "co_occurrence.xlsx"
TOP_NODES_XLSX = OUTPUT_FOLDER / "top_nodes.xlsx"
FILES_LOG_XLSX = OUTPUT_FOLDER / "files_log.xlsx"

NAME_AGGREGATION_FILENAME = "Monumenta_Indica_occurrences_long_with_name_aggregation.xlsx"
NAME_AGGREGATION_SHEET = "name_aggregation"
NAME_AGGREGATION_CANDIDATES = [
    INPUT_FOLDER / NAME_AGGREGATION_FILENAME,
    OUTPUT_FOLDER / NAME_AGGREGATION_FILENAME,
]

# Drugi etap agregacji – decyzje manualne użytkownika.
MANUAL_DECISIONS_FILENAME = "Monumenta_Indica_manual_name_decisions.csv"
MANUAL_DECISIONS_CANDIDATES = [
    SCRIPT_DIR / MANUAL_DECISIONS_FILENAME,
    INPUT_FOLDER / MANUAL_DECISIONS_FILENAME,
]

PAGE_MAX = 1100

NAME_COLUMN_ALIASES = ("Name", "nazwa")
PAGES_COLUMN_ALIASES = ("Page numbers", "numery stron")
CONTEXT_COLUMN_ALIASES = ("Context", "kontekst")


# ============================================================
# 2. FUNKCJE POMOCNICZE
# ============================================================

def normalize_column_name(col):
    value = str(col).replace("\ufeff", " ").strip().casefold()
    value = re.sub(r"\s+", " ", value)
    return value


def find_column(column_map, aliases):
    for alias in aliases:
        key = normalize_column_name(alias)
        if key in column_map:
            return column_map[key]
    return None


def clean_entity_name(value):
    if pd.isna(value):
        return None

    value = str(value).strip()
    value = re.sub(r"\s+", " ", value)

    return value or None


def clean_context(value):
    if pd.isna(value):
        return ""

    value = str(value).strip()
    value = re.sub(r"\s+", " ", value)
    return value


def expand_page_range(start, end):
    start = int(start)
    end_raw = str(end)

    if len(end_raw) < len(str(start)):
        prefix = str(start)[: len(str(start)) - len(end_raw)]
        end = int(prefix + end_raw)
    else:
        end = int(end_raw)

    if end < start:
        return [start]

    if end - start > 100:
        return [start]

    return list(range(start, end + 1))


def parse_pages(value, page_max=PAGE_MAX):
    if pd.isna(value):
        return []

    text = str(value).strip()
    if not text:
        return []

    text = text.replace(";", ",")
    text = text.replace("|", ",")
    text = re.sub(r"\s+", " ", text)

    pages = []

    for token in [t.strip() for t in text.split(",") if t.strip()]:
        token = re.sub(r"[sS]+$", "", token)
        token = token.replace("*", "")
        token = token.replace(".", "")
        token = token.strip()

        range_match = re.fullmatch(r"(\d{1,4})\s*[-–—]\s*(\d{1,4})", token)
        if range_match:
            start, end = range_match.groups()
            pages.extend(expand_page_range(start, end))
            continue

        single_match = re.search(r"\d{1,4}", token)
        if single_match:
            page = int(single_match.group())

            if 1 <= page <= page_max:
                pages.append(page)

    return sorted(set(pages))


def get_volume_id(file_path):
    stem = file_path.stem
    match = re.search(r"(\d+)", stem)

    if match:
        return int(match.group(1))

    return stem


def join_unique_text(values, sep=" | "):
    result = []
    seen = set()

    for value in values:
        if pd.isna(value):
            continue

        value = clean_entity_name(value)
        if not value or value in seen:
            continue

        seen.add(value)
        result.append(value)

    return sep.join(result)


def collect_aliases(values):
    aliases = []
    seen = set()

    for value in values:
        if pd.isna(value):
            continue

        for alias in str(value).split(" | "):
            alias = clean_entity_name(alias)
            if not alias or alias in seen:
                continue

            seen.add(alias)
            aliases.append(alias)

    return " | ".join(aliases)


def find_name_aggregation_file():
    for path in NAME_AGGREGATION_CANDIDATES:
        if path.exists():
            return path

    expected = "\n".join(f"  - {path}" for path in NAME_AGGREGATION_CANDIDATES)
    raise FileNotFoundError(
        "Nie znaleziono skoroszytu z agregacją nazw.\n"
        f"Oczekiwany plik: {NAME_AGGREGATION_FILENAME}\n"
        f"Sprawdzono:\n{expected}"
    )


def load_name_aggregation(file_path, sheet_name=NAME_AGGREGATION_SHEET):
    try:
        aggregation_df = pd.read_excel(file_path, sheet_name=sheet_name)
    except ValueError as e:
        raise ValueError(
            f"Nie znaleziono arkusza '{sheet_name}' w pliku: {file_path}"
        ) from e

    normalized_columns = {
        normalize_column_name(column): column
        for column in aggregation_df.columns
    }

    required = {"original_name", "canonical_name"}
    missing = required - set(normalized_columns)

    if missing:
        raise ValueError(
            "Arkusz agregacji nie zawiera wymaganych kolumn: "
            + ", ".join(sorted(missing))
        )

    original_col = normalized_columns["original_name"]
    canonical_col = normalized_columns["canonical_name"]

    mapping_df = aggregation_df[[original_col, canonical_col]].copy()
    mapping_df.columns = ["original_name", "canonical_name"]

    mapping_df["original_name"] = mapping_df["original_name"].apply(clean_entity_name)
    mapping_df["canonical_name"] = mapping_df["canonical_name"].apply(clean_entity_name)
    mapping_df = mapping_df.dropna(subset=["original_name"]).copy()

    mapping_df["canonical_name"] = mapping_df["canonical_name"].fillna(
        mapping_df["original_name"]
    )

    conflict_counts = (
        mapping_df
        .groupby("original_name")["canonical_name"]
        .nunique()
    )
    conflicting_names = conflict_counts[conflict_counts > 1].index.tolist()

    if conflicting_names:
        examples = ", ".join(conflicting_names[:10])
        raise ValueError(
            "W arkuszu agregacji wykryto sprzeczne mapowania dla "
            f"{len(conflicting_names)} nazw. Przykłady: {examples}"
        )

    mapping_df = mapping_df.drop_duplicates(
        subset=["original_name"],
        keep="first",
    )

    mapping = dict(
        zip(mapping_df["original_name"], mapping_df["canonical_name"])
    )

    changed_forms = mapping_df[
        mapping_df["original_name"] != mapping_df["canonical_name"]
    ]

    print(f"Wczytano agregację nazw: {file_path}")
    print(f"Liczba form w słowniku: {len(mapping):,}")
    print(
        "Liczba form z rzeczywistym nadpisaniem: "
        f"{len(changed_forms):,}"
    )
    print(
        "Do nadpisania używana jest wyłącznie kolumna 'canonical_name'; "
        "'suggested_canonical_name' jest celowo ignorowana."
    )

    return mapping


def find_manual_decisions_file():
    for path in MANUAL_DECISIONS_CANDIDATES:
        if path.exists():
            return path

    expected = "\n".join(f"  - {path}" for path in MANUAL_DECISIONS_CANDIDATES)
    raise FileNotFoundError(
        "Nie znaleziono pliku z manualnymi decyzjami agregacyjnymi.\n"
        f"Oczekiwany plik: {MANUAL_DECISIONS_FILENAME}\n"
        f"Sprawdzono:\n{expected}"
    )


def load_manual_decisions(file_path):
    """
    Wczytuje drugi etap agregacji: `name -> proper name`.

    Decyzje są stosowane dopiero do wyniku automatycznej agregacji.
    Zachowujemy także mapowania X -> X, ponieważ dokumentują ręczną
    weryfikację nazwy, nawet jeżeli nie powodują zmiany tekstowej.
    """
    decisions_df = pd.read_csv(file_path, dtype=str, encoding="utf-8-sig")

    normalized_columns = {
        normalize_column_name(column): column
        for column in decisions_df.columns
    }

    required = {"name", "proper name"}
    missing = required - set(normalized_columns)
    if missing:
        raise ValueError(
            "Plik z manualnymi decyzjami nie zawiera wymaganych kolumn: "
            + ", ".join(sorted(missing))
        )

    name_col = normalized_columns["name"]
    proper_col = normalized_columns["proper name"]

    mapping_df = decisions_df[[name_col, proper_col]].copy()
    mapping_df.columns = ["name", "proper_name"]
    mapping_df["name"] = mapping_df["name"].apply(clean_entity_name)
    mapping_df["proper_name"] = mapping_df["proper_name"].apply(clean_entity_name)
    mapping_df = mapping_df.dropna(subset=["name", "proper_name"]).copy()

    conflict_counts = (
        mapping_df
        .groupby("name")["proper_name"]
        .nunique()
    )
    conflicting_names = conflict_counts[conflict_counts > 1].index.tolist()

    if conflicting_names:
        examples = ", ".join(conflicting_names[:10])
        raise ValueError(
            "W manualnych decyzjach wykryto sprzeczne mapowania dla "
            f"{len(conflicting_names)} nazw. Przykłady: {examples}"
        )

    mapping_df = mapping_df.drop_duplicates(subset=["name"], keep="last")
    mapping = dict(zip(mapping_df["name"], mapping_df["proper_name"]))

    changed = mapping_df[mapping_df["name"] != mapping_df["proper_name"]]

    print(f"Wczytano manualne decyzje: {file_path}")
    print(f"Liczba ręcznie rozstrzygniętych nazw: {len(mapping):,}")
    print(f"Liczba decyzji zmieniających nazwę: {len(changed):,}")

    return mapping


def safe_sigma_write_html(graph, path, **kwargs):
    if Sigma is None:
        print(
            f"Pominięto eksport HTML {Path(path).name}: "
            "pakiet ipysigma nie jest zainstalowany."
        )
        return

    try:
        Sigma.write_html(graph, str(path), **kwargs)
        print(f"Zapisano HTML: {path}")
    except Exception as e:
        print(f"Nie udało się zapisać HTML {path}. Błąd: {e}")


# ============================================================
# 3. WALIDACJA ŚCIEŻEK I WCZYTANIE PLIKÓW
# ============================================================

print(f"Folder wejściowy: {INPUT_FOLDER}")

if not INPUT_FOLDER.exists():
    raise FileNotFoundError(
        "Nie znaleziono folderu z plikami Monumenta Indica:\n"
        f"  {INPUT_FOLDER}\n\n"
        "Oczekiwana struktura lokalna:\n"
        "  <repozytorium>/data/Monumenta Indica/*.xlsx"
    )

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

aggregation_path = find_name_aggregation_file()
preferred_forms = load_name_aggregation(aggregation_path)

manual_decisions_path = find_manual_decisions_file()
manual_forms = load_manual_decisions(manual_decisions_path)

excel_files = (
    list(INPUT_FOLDER.glob("*.xlsx"))
    + list(INPUT_FOLDER.glob("*.xls"))
    + list(INPUT_FOLDER.glob("*.xlsm"))
)

excel_files = sorted({p.resolve() for p in excel_files})
excel_files = [
    p
    for p in excel_files
    if not p.name.startswith("~$")
    and "name_unification_output" not in str(p)
    and "network_output" not in str(p)
    and p.resolve() != aggregation_path.resolve()
]

print(f"Liczba znalezionych plików Excel: {len(excel_files)}")
for p in excel_files:
    print(f"  - {p.name}")

all_occurrences = []
file_log = []

for file_path in tqdm(excel_files, desc="Wczytywanie plików"):
    volume_id = get_volume_id(file_path)

    try:
        xls = pd.ExcelFile(file_path)
    except Exception as e:
        file_log.append(
            {
                "file": file_path.name,
                "volume": volume_id,
                "sheet": "",
                "status": "ERROR_OPEN_FILE",
                "rows_read": 0,
                "occurrences_created": 0,
                "message": str(e),
            }
        )
        continue

    index_sheet = next(
        (s for s in xls.sheet_names if normalize_column_name(s) == "index"),
        None,
    )

    sheets_to_read = [index_sheet] if index_sheet is not None else xls.sheet_names

    for sheet_name in sheets_to_read:
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        except Exception as e:
            file_log.append(
                {
                    "file": file_path.name,
                    "volume": volume_id,
                    "sheet": sheet_name,
                    "status": "ERROR_READ_SHEET",
                    "rows_read": 0,
                    "occurrences_created": 0,
                    "message": str(e),
                }
            )
            continue

        column_map = {
            normalize_column_name(c): c
            for c in df.columns
        }

        name_col = find_column(column_map, NAME_COLUMN_ALIASES)
        pages_col = find_column(column_map, PAGES_COLUMN_ALIASES)
        context_col = find_column(column_map, CONTEXT_COLUMN_ALIASES)

        if name_col is None or pages_col is None:
            file_log.append(
                {
                    "file": file_path.name,
                    "volume": volume_id,
                    "sheet": sheet_name,
                    "status": "SKIPPED_MISSING_COLUMNS",
                    "rows_read": 0,
                    "occurrences_created": 0,
                    "message": (
                        "Wymagane kolumny: Name + Page numbers "
                        "(lub nazwa + numery stron). "
                        f"Dostępne: {', '.join(map(str, df.columns))}"
                    ),
                }
            )
            continue

        local = df.copy()
        local["name_original"] = local[name_col].apply(clean_entity_name)
        local["pages_raw"] = local[pages_col]

        if context_col is not None:
            local["context"] = local[context_col].apply(clean_context)
        else:
            local["context"] = ""

        local = local.dropna(subset=["name_original"])

        occurrence_count_before = len(all_occurrences)

        for idx, row in local.iterrows():
            pages = parse_pages(row["pages_raw"])

            for page in pages:
                all_occurrences.append(
                    {
                        "volume": volume_id,
                        "source_file": file_path.name,
                        "source_sheet": sheet_name,
                        "row_in_sheet": int(idx) + 2,
                        "page": page,
                        "name_original": row["name_original"],
                        "context": row["context"],
                    }
                )

        file_log.append(
            {
                "file": file_path.name,
                "volume": volume_id,
                "sheet": sheet_name,
                "status": "READ",
                "rows_read": len(local),
                "occurrences_created": len(all_occurrences) - occurrence_count_before,
                "message": "",
            }
        )

file_log_df = pd.DataFrame(file_log)

occurrences_df = pd.DataFrame(all_occurrences)

if occurrences_df.empty:
    diagnostics = ""
    if file_log:
        diagnostics = "\n\nDiagnostyka:\n" + "\n".join(
            f"  - {x['file']} | {x['sheet'] or '-'} | "
            f"{x['status']} | {x['message']}"
            for x in file_log
        )

    raise ValueError(
        "Nie znaleziono żadnych wystąpień bytów na stronach.\n"
        f"Folder: {INPUT_FOLDER}\n"
        f"Liczba znalezionych plików Excel: {len(excel_files)}\n"
        "Skrypt akceptuje kolumny Name + Page numbers (+ Context) "
        "oraz starsze polskie odpowiedniki nazwa + numery stron (+ kontekst)."
        f"{diagnostics}"
    )

original_unique_count = occurrences_df["name_original"].nunique()

# ------------------------------------------------------------
# ETAP 1 – automatyczna / ekspercka agregacja przygotowana wcześniej
# ------------------------------------------------------------
occurrences_df["automatic_name"] = (
    occurrences_df["name_original"]
    .map(preferred_forms)
    .fillna(occurrences_df["name_original"])
)

occurrences_df["automatic_aggregation_applied"] = (
    occurrences_df["automatic_name"] != occurrences_df["name_original"]
)

unmapped_forms = sorted(
    set(occurrences_df["name_original"].unique()) - set(preferred_forms)
)

auto_changed_occurrences = int(
    occurrences_df["automatic_aggregation_applied"].sum()
)
auto_changed_forms = occurrences_df.loc[
    occurrences_df["automatic_aggregation_applied"],
    "name_original",
].nunique()

print("\nETAP 1 – agregacja automatyczna/ekspercka:")
print(f"Unikalne formy przed agregacją: {original_unique_count:,}")
print(f"Formy faktycznie nadpisane: {auto_changed_forms:,}")
print(f"Wystąpienia z nadpisaną nazwą: {auto_changed_occurrences:,}")
print(
    "Formy spoza słownika agregacji pozostawione bez zmian: "
    f"{len(unmapped_forms):,}"
)

if unmapped_forms:
    print("Przykładowe formy spoza słownika:")
    for form in unmapped_forms[:20]:
        print(f"  - {form}")

# ------------------------------------------------------------
# ETAP 2 – manualne decyzje użytkownika
# ------------------------------------------------------------
# Kluczowe: manualne mapowanie jest przykładane do `automatic_name`,
# a nie do `name_original`. Jest więc rzeczywiście kolejnym etapem.
occurrences_df["manual_decision_applied"] = (
    occurrences_df["automatic_name"].isin(manual_forms)
)

occurrences_df["name"] = (
    occurrences_df["automatic_name"]
    .map(manual_forms)
    .fillna(occurrences_df["automatic_name"])
)

occurrences_df["manual_name_changed"] = (
    occurrences_df["name"] != occurrences_df["automatic_name"]
)

# Łączna informacja o rzeczywistej zmianie nazwy na którymkolwiek etapie.
# Nie obejmuje decyzji manualnej typu X -> X, bo taka decyzja jest kontrolą,
# ale nie zmianą formy nazewniczej.
occurrences_df["aggregation_applied"] = (
    occurrences_df["automatic_aggregation_applied"]
    | occurrences_df["manual_name_changed"]
)

manual_checked_occurrences = int(
    occurrences_df["manual_decision_applied"].sum()
)
manual_changed_occurrences = int(
    occurrences_df["manual_name_changed"].sum()
)
manual_checked_forms = occurrences_df.loc[
    occurrences_df["manual_decision_applied"],
    "automatic_name",
].nunique()
manual_changed_forms = occurrences_df.loc[
    occurrences_df["manual_name_changed"],
    "automatic_name",
].nunique()

print("\nETAP 2 – decyzje manualne:")
print(f"Ręcznie rozstrzygnięte formy obecne w danych: {manual_checked_forms:,}")
print(f"Formy faktycznie zmienione manualnie: {manual_changed_forms:,}")
print(f"Wystąpienia objęte decyzją manualną: {manual_checked_occurrences:,}")
print(f"Wystąpienia faktycznie zmienione manualnie: {manual_changed_occurrences:,}")

# Deduplikację wykonujemy dopiero po obu etapach. W przeciwnym razie
# utracilibyśmy część informacji o tym, które formy zostały połączone.
rows_before_deduplication = len(occurrences_df)

occurrences_df = (
    occurrences_df
    .groupby(["volume", "page", "name"], as_index=False, sort=False)
    .agg(
        source_file=("source_file", join_unique_text),
        source_sheet=("source_sheet", join_unique_text),
        row_in_sheet=("row_in_sheet", lambda x: ", ".join(
            map(str, sorted(set(int(v) for v in x if pd.notna(v))))
        )),
        original_names=("name_original", join_unique_text),
        automatic_names=("automatic_name", join_unique_text),
        context=("context", join_unique_text),
        automatic_aggregation_applied=("automatic_aggregation_applied", "max"),
        manual_decision_applied=("manual_decision_applied", "max"),
        manual_name_changed=("manual_name_changed", "max"),
        aggregation_applied=("aggregation_applied", "max"),
    )
)

rows_after_deduplication = len(occurrences_df)
collapsed_rows = rows_before_deduplication - rows_after_deduplication

occurrences_df = occurrences_df[
    [
        "volume",
        "source_file",
        "source_sheet",
        "row_in_sheet",
        "page",
        "original_names",
        "automatic_names",
        "name",
        "context",
        "automatic_aggregation_applied",
        "manual_decision_applied",
        "manual_name_changed",
        "aggregation_applied",
    ]
]

print("\nWYNIK PO OBU ETAPACH:")
print(f"Rekordy scalone na tej samej stronie: {collapsed_rows:,}")
print(f"Liczba wystąpień bytów: {len(occurrences_df):,}")
print(f"Liczba unikalnych bytów: {occurrences_df['name'].nunique():,}")
print(f"Liczba tomów: {occurrences_df['volume'].nunique():,}")

occurrences_df.to_excel(OCCURRENCES_XLSX, index=False)
file_log_df.to_excel(FILES_LOG_XLSX, index=False)


# ============================================================
# 4. BUDOWA RELACJI WSPÓŁWYSTĘPOWANIA
# ============================================================

relations = []

grouped = occurrences_df.groupby(["volume", "page"], sort=False)

for (volume, page), group in tqdm(grouped, desc="Budowa relacji"):
    names = sorted(group["name"].dropna().unique())

    if len(names) < 2:
        continue

    for name_1, name_2 in combinations(names, 2):
        relations.append(
            {
                "Name_1": name_1,
                "Name_2": name_2,
                "volume": volume,
                "page": page,
                "shared_page_id": f"{volume}:{page}",
            }
        )

relations_raw_df = pd.DataFrame(relations)

if relations_raw_df.empty:
    raise ValueError(
        "Nie utworzono żadnych relacji współwystępowania. "
        "Dane zostały wczytane, ale żadna strona nie zawiera co najmniej "
        "dwóch różnych bytów."
    )

print(
    "Liczba surowych relacji strona-po-stronie: "
    f"{len(relations_raw_df)}"
)


# ============================================================
# 5. AGREGACJA RELACJI
# ============================================================

co_occurrence = (
    relations_raw_df
    .groupby(["Name_1", "Name_2"], as_index=False)
    .agg(
        weight=("shared_page_id", "count"),
        shared_pages_count=("shared_page_id", "nunique"),
        volumes_count=("volume", "nunique"),
        volumes=(
            "volume",
            lambda x: ", ".join(sorted(set(map(str, x)))),
        ),
        shared_pages=(
            "shared_page_id",
            lambda x: ", ".join(sorted(set(map(str, x)))),
        ),
    )
)

co_occurrence = co_occurrence.sort_values(
    by=["weight", "Name_1", "Name_2"],
    ascending=[False, True, True],
).reset_index(drop=True)

print(f"Liczba unikalnych relacji: {len(co_occurrence)}")


# ============================================================
# 6. EKSPORT TABEL
# ============================================================

with pd.ExcelWriter(RELATIONS_XLSX, engine="openpyxl") as writer:
    co_occurrence.to_excel(
        writer,
        sheet_name="relations_aggregated",
        index=False,
    )
    relations_raw_df.to_excel(
        writer,
        sheet_name="relations_raw_pages",
        index=False,
    )

co_occurrence.to_excel(CO_OCCURRENCE_XLSX, index=False)

print(f"Zapisano tabelę relacji: {RELATIONS_XLSX}")
print(f"Zapisano tabelę wystąpień: {OCCURRENCES_XLSX}")
print(f"Zapisano log plików: {FILES_LOG_XLSX}")


# ============================================================
# 7. BUDOWA GRAFU NETWORKX
# ============================================================

G = nx.Graph()

node_stats = (
    occurrences_df
    .groupby("name", as_index=False)
    .agg(
        occurrence_count=("page", "count"),
        page_count=("page", "nunique"),
        volume_count=("volume", "nunique"),
        aliases=("original_names", collect_aliases),
        aggregated_page_count=("aggregation_applied", "sum"),
        automatic_aggregated_page_count=("automatic_aggregation_applied", "sum"),
        manual_reviewed_page_count=("manual_decision_applied", "sum"),
        manual_changed_page_count=("manual_name_changed", "sum"),
    )
)

node_stats_dict = node_stats.set_index("name").to_dict(orient="index")

for name, stats in node_stats_dict.items():
    G.add_node(
        name,
        label=name,
        aliases=stats["aliases"],
        occurrence_count=int(stats["occurrence_count"]),
        page_count=int(stats["page_count"]),
        volume_count=int(stats["volume_count"]),
        aggregated_page_count=int(stats["aggregated_page_count"]),
        automatic_aggregated_page_count=int(
            stats["automatic_aggregated_page_count"]
        ),
        manual_reviewed_page_count=int(
            stats["manual_reviewed_page_count"]
        ),
        manual_changed_page_count=int(
            stats["manual_changed_page_count"]
        ),
    )

for _, row in tqdm(
    co_occurrence.iterrows(),
    total=len(co_occurrence),
    desc="Dodawanie krawędzi",
):
    G.add_edge(
        row["Name_1"],
        row["Name_2"],
        weight=int(row["weight"]),
        shared_pages_count=int(row["shared_pages_count"]),
        volumes_count=int(row["volumes_count"]),
        volumes=row["volumes"],
        shared_pages=row["shared_pages"],
    )

print(f"Liczba węzłów w grafie: {G.number_of_nodes()}")
print(f"Liczba krawędzi w grafie: {G.number_of_edges()}")


# ============================================================
# 8. METRYKI SIECIOWE
# ============================================================

degree_dict = dict(G.degree())
weighted_degree_dict = dict(G.degree(weight="weight"))

nx.set_node_attributes(G, degree_dict, name="degree")
nx.set_node_attributes(
    G,
    weighted_degree_dict,
    name="weighted_degree",
)

if G.number_of_edges() > 0:
    pagerank = nx.pagerank(G, weight="weight")
    nx.set_node_attributes(G, pagerank, name="pagerank")

    try:
        communities = nx.community.louvain_communities(
            G,
            weight="weight",
            seed=42,
        )

        louvain_dict = {}

        for i, community in enumerate(communities):
            for node in community:
                louvain_dict[node] = str(i)

        nx.set_node_attributes(
            G,
            louvain_dict,
            name="louvain",
        )

    except Exception as e:
        print(
            "Nie udało się policzyć Louvain w NetworkX. "
            f"Błąd: {e}"
        )


# ============================================================
# 9. PODGLĄD I EKSPORT HTML
# ============================================================

if Sigma is not None and G.number_of_nodes() > 0:
    try:
        Sigma(
            G,
            node_color=(
                "louvain"
                if any(
                    "louvain" in data
                    for _, data in G.nodes(data=True)
                )
                else None
            ),
            node_size="degree",
            node_label_size="degree",
        )
    except Exception as e:
        print(f"Nie udało się utworzyć podglądu Sigma. Błąd: {e}")

if any("louvain" in data for _, data in G.nodes(data=True)):
    safe_sigma_write_html(
        G,
        HTML_PATH,
        fullscreen=True,
        node_metrics=["louvain"],
        node_color="louvain",
        node_size="degree",
        node_size_range=(3, 20),
        max_categorical_colors=30,
        default_edge_type="curve",
        default_node_label_size=14,
        node_border_color_from="node",
    )
else:
    print(
        "Pominięto wersję HTML z Louvain, "
        "ponieważ atrybut louvain nie został utworzony."
    )

if any("pagerank" in data for _, data in G.nodes(data=True)):
    safe_sigma_write_html(
        G,
        HTML_PAGERANK_PATH,
        fullscreen=True,
        node_color="pagerank",
        node_size="degree",
        node_size_range=(3, 20),
        max_categorical_colors=30,
        default_edge_type="curve",
        default_node_label_size=14,
        node_border_color_from="node",
    )
else:
    print(
        "Pominięto wersję HTML z PageRank, "
        "ponieważ atrybut pagerank nie został utworzony."
    )


# ============================================================
# 10. EKSPORT GRAPHML
# ============================================================

nx.write_graphml(G, GRAPHML_PATH)
print(f"Zapisano GraphML: {GRAPHML_PATH}")


# ============================================================
# 11. KONTROLA WYNIKÓW
# ============================================================

top_nodes = (
    pd.DataFrame(
        [
            {
                "name": node,
                "aliases": data.get("aliases", ""),
                "degree": data.get("degree", 0),
                "weighted_degree": data.get("weighted_degree", 0),
                "pagerank": data.get("pagerank"),
                "occurrence_count": data.get("occurrence_count", 0),
                "page_count": data.get("page_count", 0),
                "volume_count": data.get("volume_count", 0),
                "aggregated_page_count": data.get("aggregated_page_count", 0),
                "automatic_aggregated_page_count": data.get(
                    "automatic_aggregated_page_count", 0
                ),
                "manual_reviewed_page_count": data.get(
                    "manual_reviewed_page_count", 0
                ),
                "manual_changed_page_count": data.get(
                    "manual_changed_page_count", 0
                ),
                "louvain": data.get("louvain"),
            }
            for node, data in G.nodes(data=True)
        ]
    )
    .sort_values("weighted_degree", ascending=False)
    .reset_index(drop=True)
)

top_nodes.to_excel(TOP_NODES_XLSX, index=False)

print("\nNajsilniejsze relacje:")
print(co_occurrence.head(20))

print("\nNajważniejsze węzły według stopnia ważonego:")
print(top_nodes.head(20))

print("\nPodsumowanie:")
print(f"Pliki Excel: {len(excel_files)}")
print(f"Tomy: {occurrences_df['volume'].nunique()}")
print(f"Wystąpienia: {len(occurrences_df)}")
print(f"Byty: {occurrences_df['name'].nunique()}")
print(f"Relacje unikalne: {len(co_occurrence)}")
print(f"Węzły grafu: {G.number_of_nodes()}")
print(f"Krawędzie grafu: {G.number_of_edges()}")
