from pathlib import Path
import re
import pandas as pd
from tqdm import tqdm


# ============================================================
# 1. KONFIGURACJA
# ============================================================

INPUT_FOLDER = Path(r"C:\Users\pracownik\Documents\robert-maryks\data\Monumenta Indica")

OUTPUT_FOLDER = INPUT_FOLDER / "name_unification_output"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

OUTPUT_XLSX = OUTPUT_FOLDER / "Monumenta_Indica_name_unification_template.xlsx"

PAGE_MAX = 999

EXPECTED_NAME_COL = "nazwa"
EXPECTED_PAGES_COL = "numery stron"
EXPECTED_CONTEXT_COL = "kontekst"


# ============================================================
# 2. FUNKCJE POMOCNICZE
# ============================================================

def normalize_column_name(col):
    return str(col).strip().lower()


def clean_name(value):
    """
    Minimalne czyszczenie formy nazewniczej.
    Nie dokonujemy jeszcze agregacji historycznej ani identyfikacji osób.
    """
    if pd.isna(value):
        return None

    value = str(value).strip()
    value = re.sub(r"\s+", " ", value)

    if value == "":
        return None

    return value


def make_name_key(value):
    """
    Klucz techniczny do grupowania identycznych form po lekkiej normalizacji.
    Nie usuwa przecinków ani nie przestawia szyku nazwisk.
    """
    if value is None or pd.isna(value):
        return None

    value = str(value).strip()
    value = re.sub(r"\s+", " ", value)
    value = value.casefold()

    return value


def clean_context(value):
    if pd.isna(value):
        return ""

    value = str(value).strip()
    value = re.sub(r"\s+", " ", value)

    return value


def expand_page_range(start, end):
    """
    Rozwija zakresy:
    423-426 -> 423, 424, 425, 426
    423-26  -> 423, 424, 425, 426
    """
    start = int(start)
    end_raw = str(end)

    if len(end_raw) < len(str(start)):
        prefix = str(start)[:len(str(start)) - len(end_raw)]
        end = int(prefix + end_raw)
    else:
        end = int(end_raw)

    if end < start:
        return [start]

    if end - start > 100:
        return [start]

    return list(range(start, end + 1))


def parse_pages(value, page_max=PAGE_MAX):
    """
    Parsuje kolumnę 'numery stron'.

    Obsługuje m.in.:
    - 44, 231, 473
    - 423-26
    - 423-426
    - 215s
    - 546ss
    - 23*
    """
    if pd.isna(value):
        return []

    text = str(value).strip()

    if text == "":
        return []

    text = text.replace(";", ",")
    text = text.replace("|", ",")
    text = re.sub(r"\s+", " ", text)

    pages = []

    tokens = [t.strip() for t in text.split(",") if t.strip()]

    for token in tokens:
        token = token.strip()

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
    """
    Próbuje wydobyć numer tomu z nazwy pliku, np.:
    monumenta_indica_index_18.xlsx -> 18

    Jeżeli nie znajdzie numeru, używa nazwy pliku bez rozszerzenia.
    """
    stem = file_path.stem

    match = re.search(r"(\d+)", stem)

    if match:
        return int(match.group(1))

    return stem


def join_unique(values, sep=", ", limit=None):
    """
    Łączy unikalne wartości tekstowe.
    """
    cleaned = []

    for v in values:
        if pd.isna(v):
            continue

        v = str(v).strip()

        if v:
            cleaned.append(v)

    unique_values = sorted(set(cleaned))

    if limit is not None and len(unique_values) > limit:
        shown = unique_values[:limit]
        return sep.join(shown) + f" ... [+{len(unique_values) - limit}]"

    return sep.join(unique_values)


def join_unique_numbers(values, sep=", "):
    nums = []

    for v in values:
        if pd.isna(v):
            continue

        try:
            nums.append(int(v))
        except Exception:
            pass

    return sep.join(map(str, sorted(set(nums))))


def choose_display_name(names):
    """
    Dla jednej znormalizowanej formy wybiera najczęściej występujący zapis.
    Przy remisie bierze alfabetycznie pierwszy.
    """
    counts = pd.Series(names).value_counts()

    if counts.empty:
        return ""

    max_count = counts.max()
    candidates = sorted(counts[counts == max_count].index.tolist())

    return candidates[0]


# ============================================================
# 3. WCZYTANIE PLIKÓW
# ============================================================

excel_files = (
    list(INPUT_FOLDER.glob("*.xlsx")) +
    list(INPUT_FOLDER.glob("*.xls")) +
    list(INPUT_FOLDER.glob("*.xlsm"))
)

excel_files = [
    p for p in excel_files
    if not p.name.startswith("~$")
    and "name_unification_output" not in str(p)
    and "network_output" not in str(p)
]

print(f"Liczba znalezionych plików Excel: {len(excel_files)}")

rows = []
file_log = []

for file_path in tqdm(excel_files, desc="Wczytywanie indeksów"):
    volume_id = get_volume_id(file_path)

    try:
        xls = pd.ExcelFile(file_path)
    except Exception as e:
        file_log.append({
            "file": file_path.name,
            "volume": volume_id,
            "sheet": "",
            "status": "ERROR_OPEN_FILE",
            "rows_read": 0,
            "message": str(e)
        })
        continue

    for sheet_name in xls.sheet_names:
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        except Exception as e:
            file_log.append({
                "file": file_path.name,
                "volume": volume_id,
                "sheet": sheet_name,
                "status": "ERROR_READ_SHEET",
                "rows_read": 0,
                "message": str(e)
            })
            continue

        normalized_cols = [normalize_column_name(c) for c in df.columns]
        column_map = dict(zip(normalized_cols, df.columns))

        if EXPECTED_NAME_COL not in column_map:
            file_log.append({
                "file": file_path.name,
                "volume": volume_id,
                "sheet": sheet_name,
                "status": "SKIPPED_NO_NAME_COLUMN",
                "rows_read": 0,
                "message": "Brak kolumny 'nazwa'"
            })
            continue

        if EXPECTED_PAGES_COL not in column_map:
            file_log.append({
                "file": file_path.name,
                "volume": volume_id,
                "sheet": sheet_name,
                "status": "SKIPPED_NO_PAGES_COLUMN",
                "rows_read": 0,
                "message": "Brak kolumny 'numery stron'"
            })
            continue

        name_col = column_map[EXPECTED_NAME_COL]
        pages_col = column_map[EXPECTED_PAGES_COL]
        context_col = column_map.get(EXPECTED_CONTEXT_COL, None)

        local = df.copy()
        local["name_original"] = local[name_col].apply(clean_name)
        local["name_key"] = local["name_original"].apply(make_name_key)
        local["pages_raw"] = local[pages_col]

        if context_col is not None:
            local["context"] = local[context_col].apply(clean_context)
        else:
            local["context"] = ""

        local = local.dropna(subset=["name_original", "name_key"])

        for idx, row in local.iterrows():
            pages = parse_pages(row["pages_raw"])

            rows.append({
                "volume": volume_id,
                "source_file": file_path.name,
                "source_sheet": sheet_name,
                "row_in_sheet": int(idx) + 2,
                "name_original": row["name_original"],
                "name_key": row["name_key"],
                "pages_raw": row["pages_raw"],
                "pages_parsed": ", ".join(map(str, pages)),
                "page_occurrence_count_in_row": len(pages),
                "context": row["context"]
            })

        file_log.append({
            "file": file_path.name,
            "volume": volume_id,
            "sheet": sheet_name,
            "status": "READ",
            "rows_read": len(local),
            "message": ""
        })


names_df = pd.DataFrame(rows)
file_log_df = pd.DataFrame(file_log)

if names_df.empty:
    raise ValueError("Nie wczytano żadnych danych. Sprawdź, czy arkusze zawierają kolumny 'nazwa' oraz 'numery stron'.")

print(f"Liczba wczytanych rekordów indeksowych: {len(names_df)}")
print(f"Liczba unikalnych form nazewniczych: {names_df['name_key'].nunique()}")


# ============================================================
# 4. TABELA DŁUGA: FORMA × TOM × STRONA
# ============================================================

occurrence_rows = []

for _, row in names_df.iterrows():
    pages = parse_pages(row["pages_raw"])

    if not pages:
        occurrence_rows.append({
            "volume": row["volume"],
            "source_file": row["source_file"],
            "source_sheet": row["source_sheet"],
            "row_in_sheet": row["row_in_sheet"],
            "name_original": row["name_original"],
            "name_key": row["name_key"],
            "page": pd.NA,
            "context": row["context"]
        })
    else:
        for page in pages:
            occurrence_rows.append({
                "volume": row["volume"],
                "source_file": row["source_file"],
                "source_sheet": row["source_sheet"],
                "row_in_sheet": row["row_in_sheet"],
                "name_original": row["name_original"],
                "name_key": row["name_key"],
                "page": page,
                "context": row["context"]
            })

occurrences_long = pd.DataFrame(occurrence_rows)

print(f"Liczba wystąpień forma–strona: {len(occurrences_long)}")


# ============================================================
# 5. AGREGACJA FORM NAZEWNICZYCH
# ============================================================

grouped = names_df.groupby("name_key", dropna=False)

forms_summary = grouped.agg(
    form_record_count=("name_original", "count"),
    page_occurrence_count=("page_occurrence_count_in_row", "sum"),
    volumes_count=("volume", "nunique"),
    volumes=("volume", join_unique_numbers),
    source_files=("source_file", join_unique),
    source_sheets=("source_sheet", join_unique),
    raw_page_strings=("pages_raw", lambda x: join_unique(x, sep=" | ", limit=30)),
    contexts=("context", lambda x: join_unique(x, sep=" | ", limit=10))
).reset_index()

display_names = (
    names_df
    .groupby("name_key")["name_original"]
    .apply(choose_display_name)
    .reset_index(name="name_form")
)

variant_names = (
    names_df
    .groupby("name_key")["name_original"]
    .apply(lambda x: join_unique(x, sep=" | "))
    .reset_index(name="observed_variants")
)

forms_summary = forms_summary.merge(display_names, on="name_key", how="left")
forms_summary = forms_summary.merge(variant_names, on="name_key", how="left")

forms_summary = forms_summary[
    [
        "name_form",
        "name_key",
        "observed_variants",
        "form_record_count",
        "page_occurrence_count",
        "volumes_count",
        "volumes",
        "source_files",
        "source_sheets",
        "raw_page_strings",
        "contexts"
    ]
]

# Kolumny dla eksperta
forms_summary.insert(3, "canonical_name_manual", "")
forms_summary.insert(4, "entity_type_manual", "")
forms_summary.insert(5, "same_as_manual", "")
forms_summary.insert(6, "exclude_from_network_manual", "")
forms_summary.insert(7, "reviewer_notes", "")

forms_alphabetical = forms_summary.sort_values(
    by=["name_form", "page_occurrence_count", "form_record_count"],
    ascending=[True, False, False]
).reset_index(drop=True)

forms_by_frequency = forms_summary.sort_values(
    by=["page_occurrence_count", "form_record_count", "name_form"],
    ascending=[False, False, True]
).reset_index(drop=True)


# ============================================================
# 6. AGREGACJA FORMA × TOM
# ============================================================

forms_by_volume = (
    names_df
    .groupby(["name_key", "volume"], as_index=False)
    .agg(
        name_form=("name_original", choose_display_name),
        form_record_count=("name_original", "count"),
        page_occurrence_count=("page_occurrence_count_in_row", "sum"),
        source_files=("source_file", join_unique),
        raw_page_strings=("pages_raw", lambda x: join_unique(x, sep=" | ", limit=20)),
        contexts=("context", lambda x: join_unique(x, sep=" | ", limit=5))
    )
)

forms_by_volume = forms_by_volume.sort_values(
    by=["name_form", "volume"]
).reset_index(drop=True)


# ============================================================
# 7. DODATKOWE KONTROLE
# ============================================================

possible_duplicates = (
    forms_summary
    .assign(
        simplified_for_check=lambda df: (
            df["name_form"]
            .str.casefold()
            .str.replace(r"[^a-ząćęłńóśźż0-9 ]", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
    )
    .groupby("simplified_for_check", as_index=False)
    .agg(
        forms_count=("name_form", "nunique"),
        forms=("name_form", lambda x: join_unique(x, sep=" | ")),
        total_page_occurrence_count=("page_occurrence_count", "sum"),
        volumes=("volumes", lambda x: join_unique(x, sep=" | "))
    )
)

possible_duplicates = possible_duplicates[
    possible_duplicates["forms_count"] > 1
].sort_values(
    by=["forms_count", "total_page_occurrence_count"],
    ascending=[False, False]
).reset_index(drop=True)


# ============================================================
# 8. EKSPORT DO EXCELA
# ============================================================

with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
    forms_alphabetical.to_excel(writer, sheet_name="forms_alphabetical", index=False)
    forms_by_frequency.to_excel(writer, sheet_name="forms_by_frequency", index=False)
    forms_by_volume.to_excel(writer, sheet_name="forms_by_volume", index=False)
    occurrences_long.to_excel(writer, sheet_name="occurrences_long", index=False)
    possible_duplicates.to_excel(writer, sheet_name="possible_duplicates", index=False)
    file_log_df.to_excel(writer, sheet_name="files_log", index=False)

print(f"Zapisano plik: {OUTPUT_XLSX}")


# ============================================================
# 9. PODSTAWOWE PODSUMOWANIE
# ============================================================

print("\nPodsumowanie:")
print(f"Liczba plików Excel: {len(excel_files)}")
print(f"Liczba rekordów indeksowych: {len(names_df)}")
print(f"Liczba unikalnych form nazewniczych: {forms_summary['name_key'].nunique()}")
print(f"Liczba wystąpień forma–strona: {len(occurrences_long)}")

print("\nNajczęstsze formy nazewnicze:")
print(
    forms_by_frequency[
        [
            "name_form",
            "form_record_count",
            "page_occurrence_count",
            "volumes_count",
            "volumes"
        ]
    ].head(30)
)























