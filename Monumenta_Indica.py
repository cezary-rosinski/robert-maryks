from pathlib import Path
from itertools import combinations
import re

import pandas as pd
import networkx as nx
from tqdm import tqdm
from ipysigma import Sigma


# ============================================================
# 1. KONFIGURACJA
# ============================================================

INPUT_FOLDER = Path(r"C:\Users\pracownik\Documents\robert-maryks\data\Monumenta Indica")

OUTPUT_FOLDER = INPUT_FOLDER / "network_output"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

RELATIONS_XLSX = OUTPUT_FOLDER / "Monumenta_Indica_relations_final.xlsx"
OCCURRENCES_XLSX = OUTPUT_FOLDER / "Monumenta_Indica_occurrences_long.xlsx"
GRAPHML_PATH = OUTPUT_FOLDER / "Monumenta_Indica.graphml"
HTML_PATH = OUTPUT_FOLDER / "Monumenta_Indica_network.html"
HTML_PAGERANK_PATH = OUTPUT_FOLDER / "Monumenta_Indica_network_pagerank.html"

PAGE_MAX = 999  # cały tom ma mniej niż 1000 stron; zmień, jeśli któryś tom ma więcej


# ============================================================
# 2. FUNKCJE POMOCNICZE
# ============================================================

def normalize_column_name(col):
    """
    Normalizuje nazwy kolumn, żeby uniknąć problemów ze spacjami,
    wielkością liter itd.
    """
    return str(col).strip().lower()


def clean_entity_name(value):
    """
    Czyści nazwę bytu.
    """
    if pd.isna(value):
        return None

    value = str(value).strip()
    value = re.sub(r"\s+", " ", value)

    if value == "":
        return None

    return value


def expand_page_range(start, end):
    """
    Rozwija zakresy stron.
    Przykłady:
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
        # zabezpieczenie przed błędnym OCR-em lub źle odczytanym zakresem
        return [start]

    return list(range(start, end + 1))


def parse_pages(value, page_max=PAGE_MAX):
    """
    Parsuje zawartość kolumny 'numery stron'.

    Obsługuje m.in.:
    - '691, 707'
    - '423-26'
    - '546ss'
    - '215s'
    - '23*'

    Zwraca posortowaną listę unikalnych numerów stron.
    """
    if pd.isna(value):
        return []

    text = str(value).strip()

    if text == "":
        return []

    # ujednolicenie separatorów
    text = text.replace(";", ",")
    text = text.replace("|", ",")
    text = re.sub(r"\s+", " ", text)

    pages = []

    tokens = [t.strip() for t in text.split(",") if t.strip()]

    for token in tokens:
        token = token.strip()

        # usuwamy typowe sufiksy indeksowe: s, ss, *, kropki
        token = re.sub(r"[sS]+$", "", token)
        token = token.replace("*", "")
        token = token.replace(".", "")
        token = token.strip()

        # zakres stron, np. 423-26 albo 423-426
        range_match = re.fullmatch(r"(\d{1,4})\s*[-–—]\s*(\d{1,4})", token)
        if range_match:
            start, end = range_match.groups()
            expanded = expand_page_range(start, end)
            pages.extend(expanded)
            continue

        # pojedynczy numer strony
        single_match = re.search(r"\d{1,4}", token)
        if single_match:
            page = int(single_match.group())

            if 1 <= page <= page_max:
                pages.append(page)

    return sorted(set(pages))


def get_volume_id(file_path):
    """
    Tworzy identyfikator tomu na podstawie nazwy pliku.
    """
    return file_path.stem


# ============================================================
# 3. WCZYTANIE WSZYSTKICH PLIKÓW EXCEL
# ============================================================

excel_files = (
    list(INPUT_FOLDER.glob("*.xlsx")) +
    list(INPUT_FOLDER.glob("*.xls")) +
    list(INPUT_FOLDER.glob("*.xlsm"))
)

# Pomijamy pliki tymczasowe Excela
excel_files = [p for p in excel_files if not p.name.startswith("~$")]

print(f"Liczba znalezionych plików Excel: {len(excel_files)}")

all_occurrences = []

for file_path in tqdm(excel_files, desc="Wczytywanie plików"):
    volume_id = get_volume_id(file_path)

    try:
        xls = pd.ExcelFile(file_path)
    except Exception as e:
        print(f"Nie udało się otworzyć pliku: {file_path.name}. Błąd: {e}")
        continue

    for sheet_name in xls.sheet_names:
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        except Exception as e:
            print(f"Nie udało się wczytać arkusza {sheet_name} w pliku {file_path.name}. Błąd: {e}")
            continue

        original_columns = list(df.columns)
        normalized_columns = [normalize_column_name(c) for c in original_columns]
        column_map = dict(zip(normalized_columns, original_columns))

        if "nazwa" not in column_map or "numery stron" not in column_map:
            continue

        name_col = column_map["nazwa"]
        pages_col = column_map["numery stron"]

        local = df[[name_col, pages_col]].copy()
        local.columns = ["name", "pages_raw"]

        local["name"] = local["name"].apply(clean_entity_name)
        local = local.dropna(subset=["name"])

        for _, row in local.iterrows():
            name = row["name"]
            pages = parse_pages(row["pages_raw"])

            for page in pages:
                all_occurrences.append({
                    "volume": volume_id,
                    "source_file": file_path.name,
                    "source_sheet": sheet_name,
                    "page": page,
                    "name": name
                })


occurrences_df = pd.DataFrame(all_occurrences)

if occurrences_df.empty:
    raise ValueError("Nie znaleziono żadnych wystąpień. Sprawdź nazwy kolumn: 'nazwa' oraz 'numery stron'.")

# Usuwamy duplikaty: ten sam byt na tej samej stronie w tym samym tomie
occurrences_df = occurrences_df.drop_duplicates(
    subset=["volume", "page", "name"]
).reset_index(drop=True)

print(f"Liczba wystąpień bytów na stronach: {len(occurrences_df)}")
print(f"Liczba unikalnych bytów: {occurrences_df['name'].nunique()}")
print(f"Liczba tomów: {occurrences_df['volume'].nunique()}")


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
        relations.append({
            "Name_1": name_1,
            "Name_2": name_2,
            "volume": volume,
            "page": page,
            "shared_page_id": f"{volume}:{page}"
        })

relations_raw_df = pd.DataFrame(relations)

if relations_raw_df.empty:
    raise ValueError("Nie utworzono żadnych relacji. Prawdopodobnie strony nie mają współwystępujących bytów.")

print(f"Liczba surowych relacji strona-po-stronie: {len(relations_raw_df)}")


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
        volumes=("volume", lambda x: ", ".join(sorted(set(map(str, x))))),
        shared_pages=("shared_page_id", lambda x: ", ".join(sorted(set(map(str, x)))))
    )
)

co_occurrence = co_occurrence.sort_values(
    by=["weight", "Name_1", "Name_2"],
    ascending=[False, True, True]
).reset_index(drop=True)

print(f"Liczba unikalnych relacji: {len(co_occurrence)}")


# ============================================================
# 6. EKSPORT TABEL
# ============================================================

with pd.ExcelWriter(RELATIONS_XLSX, engine="openpyxl") as writer:
    co_occurrence.to_excel(writer, sheet_name="relations_aggregated", index=False)
    relations_raw_df.to_excel(writer, sheet_name="relations_raw_pages", index=False)

occurrences_df.to_excel(OCCURRENCES_XLSX, index=False)

print(f"Zapisano tabelę relacji: {RELATIONS_XLSX}")
print(f"Zapisano tabelę wystąpień: {OCCURRENCES_XLSX}")


# ============================================================
# 7. BUDOWA GRAFU NETWORKX
# ============================================================

G = nx.Graph()

# Atrybuty węzłów: liczba wystąpień, liczba stron, liczba tomów
node_stats = (
    occurrences_df
    .groupby("name", as_index=False)
    .agg(
        occurrence_count=("page", "count"),
        page_count=("page", "nunique"),
        volume_count=("volume", "nunique")
    )
)

node_stats_dict = node_stats.set_index("name").to_dict(orient="index")

for name, stats in node_stats_dict.items():
    G.add_node(
        name,
        label=name,
        occurrence_count=int(stats["occurrence_count"]),
        page_count=int(stats["page_count"]),
        volume_count=int(stats["volume_count"])
    )

for _, row in tqdm(co_occurrence.iterrows(), total=len(co_occurrence), desc="Dodawanie krawędzi"):
    G.add_edge(
        row["Name_1"],
        row["Name_2"],
        weight=int(row["weight"]),
        shared_pages_count=int(row["shared_pages_count"]),
        volumes_count=int(row["volumes_count"]),
        volumes=row["volumes"],
        shared_pages=row["shared_pages"]
    )

print(f"Liczba węzłów w grafie: {G.number_of_nodes()}")
print(f"Liczba krawędzi w grafie: {G.number_of_edges()}")


# ============================================================
# 8. METRYKI SIECIOWE
# ============================================================

degree_dict = dict(G.degree())
weighted_degree_dict = dict(G.degree(weight="weight"))

nx.set_node_attributes(G, degree_dict, name="degree")
nx.set_node_attributes(G, weighted_degree_dict, name="weighted_degree")

if G.number_of_edges() > 0:
    pagerank = nx.pagerank(G, weight="weight")
    nx.set_node_attributes(G, pagerank, name="pagerank")

    try:
        communities = nx.community.louvain_communities(G, weight="weight", seed=42)
        louvain_dict = {}

        for i, community in enumerate(communities):
            for node in community:
                louvain_dict[node] = str(i)

        nx.set_node_attributes(G, louvain_dict, name="louvain")

    except Exception as e:
        print(f"Nie udało się policzyć Louvain w NetworkX. Błąd: {e}")


# ============================================================
# 9. PODGLĄD W JUPYTERZE
# ============================================================

Sigma(
    G,
    node_color="louvain" if "louvain" in next(iter(G.nodes(data=True)))[1] else None,
    node_size="degree",
    node_label_size="degree"
)


# ============================================================
# 10. EKSPORT DO HTML — WERSJA Z LOUVAIN
# ============================================================

Sigma.write_html(
    G,
    str(HTML_PATH),
    fullscreen=True,
    node_metrics=["louvain"],
    node_color="louvain",
    node_size="degree",
    node_size_range=(3, 20),
    max_categorical_colors=30,
    default_edge_type="curve",
    default_node_label_size=14,
    node_border_color_from="node"
)

print(f"Zapisano HTML: {HTML_PATH}")


# ============================================================
# 11. EKSPORT DO HTML — WERSJA Z PAGERANK
# ============================================================

Sigma.write_html(
    G,
    str(HTML_PAGERANK_PATH),
    fullscreen=True,
    node_color="pagerank",
    node_size="degree",
    node_size_range=(3, 20),
    max_categorical_colors=30,
    default_edge_type="curve",
    default_node_label_size=14,
    node_border_color_from="node"
)

print(f"Zapisano HTML PageRank: {HTML_PAGERANK_PATH}")


# ============================================================
# 12. EKSPORT DO GRAPHML
# ============================================================

nx.write_graphml(G, GRAPHML_PATH)

print(f"Zapisano GraphML: {GRAPHML_PATH}")


# ============================================================
# 13. SZYBKA KONTROLA WYNIKÓW
# ============================================================

print("\nNajsilniejsze relacje:")
print(co_occurrence.head(20))

print("\nNajważniejsze węzły według stopnia ważonego:")
top_nodes = (
    pd.DataFrame([
        {
            "name": node,
            "degree": data.get("degree", 0),
            "weighted_degree": data.get("weighted_degree", 0),
            "pagerank": data.get("pagerank", None),
            "occurrence_count": data.get("occurrence_count", 0),
            "page_count": data.get("page_count", 0),
            "volume_count": data.get("volume_count", 0)
        }
        for node, data in G.nodes(data=True)
    ])
    .sort_values("weighted_degree", ascending=False)
)

print(top_nodes.head(20))

co_occurrence.to_excel('data/Monumenta Indica/network_output/co_occurrence.xlsx', index=False)
top_nodes.to_excel('data/Monumenta Indica/network_output/top_nodes.xlsx', index=False)












