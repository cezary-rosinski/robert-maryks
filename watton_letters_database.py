import sys
# sys.path.insert(1, 'D:\IBL\Documents\IBL-PAN-Python')
sys.path.insert(1, 'C:/Users/Cezary/Documents/IBL-PAN-Python')
import pandas as pd
from rdflib import Graph, Namespace, URIRef, Literal, BNode
from rdflib.namespace import RDF, RDFS, XSD, FOAF, OWL
import datetime
import regex as re
from my_functions import gsheet_to_df
from ast import literal_eval
from collections import defaultdict

#%% --- LOAD ---

df_test = gsheet_to_df('1kxnFmIx1-DW2oIpb67l-uTSWcEzn-k9yykLJ1TE1TKQ', 'Sheet1')

df_texts = gsheet_to_df('1kxnFmIx1-DW2oIpb67l-uTSWcEzn-k9yykLJ1TE1TKQ', 'texts')
texts_ids = df_texts['Work ID'].to_list()
df_people = gsheet_to_df('1M2gc-8cGZ8gh8TTnm4jl430bL4tccdri9nw-frUWYLQ', 'people')
df_places = gsheet_to_df('1M2gc-8cGZ8gh8TTnm4jl430bL4tccdri9nw-frUWYLQ', 'places')
df_events = gsheet_to_df('1M2gc-8cGZ8gh8TTnm4jl430bL4tccdri9nw-frUWYLQ', 'events')
df_genres = gsheet_to_df('1M2gc-8cGZ8gh8TTnm4jl430bL4tccdri9nw-frUWYLQ', 'genres')

#%%

test = df_test['Historical Figures Mentioned'].to_list()
test = [e for e in test if isinstance(e, str)]
test = [ele for sup in [[re.split(f"[$]", el) for el in e.split('|')] for e in test] for ele in sup]
test = [[el.strip() for el in e] for e in test]
test = [[el.split('@') for el in e] for e in test]

test_df = pd.DataFrame(test)

[re.split(f"[@$]", e) for e in test[0].split('|')]

names = []
for e in test:
    if len(e) == 3:
        names.append((e[0], e[1]))
    elif len(e) == 2:
        names.append((e[0], None))
        
df_names = pd.DataFrame(names).drop_duplicates()


text = """Paolo Sarpi@https://www.wikidata.org/wiki/Q1349158$Maestro Paolo| Fulgenzio Micanzio@https://www.wikidata.org/wiki/Q737310$Padre Fulgentio| Pope Paul V@https://www.wikidata.org/wiki/Q132711$Pope| Henry IV of France@https://www.wikidata.org/wiki/Q936976$the French king| Giovanni Diodati@https://www.wikidata.org/wiki/Q692115$Giovanni Diodati| Pierre Cotton@https://www.wikidata.org/wiki/Q2372712$Father Cotton| Jean Bochart de Champigny@https://www.wikidata.org/wiki/Q3170806$the French ambassador; $Monsieur de Champigni| Antonio Foscarini@https://www.wikidata.org/wiki/Q602038$the Venetian Ambassador| Carlo Emanuele I@https://www.wikidata.org/wiki/Q318091$the Duke of Savoye| Robert Bellarmine@https://www.wikidata.org/wiki/Q298664$Bellarmin| Giovanni Dolfin@https://www.wikidata.org/wiki/Q3107155$Cardinal Delfini| Papalini$Papalini| """

def parse_people(s: str) -> dict:
    people = {}

    # 1. rozdziel osoby po '|'
    chunks = [c.strip() for c in s.split('|') if c.strip()]

    for chunk in chunks:
        # przypadek z wikidatą
        if '@' in chunk:
            base, rest = chunk.split('@', 1)
            if '$' in rest:
                wikidata, forms_str = rest.split('$', 1)
            else:
                wikidata, forms_str = rest, ""
        else:
            # bez identyfikatora wikidaty (np. "Papalini$Papalini")
            if '$' in chunk:
                base, forms_str = chunk.split('$', 1)
            else:
                base, forms_str = chunk, ""
            wikidata = None

        # 2. rozdziel użycia w tekście po ';'
        forms = []
        for f in forms_str.split(';'):
            f = f.strip()
            if not f:
                continue
            if f.startswith('$'):
                f = f[1:].strip()
            forms.append(f)

        people[base.strip()] = {
            "wikidata": wikidata.strip() if wikidata else None,
            "forms": forms
        }

    return people

people_dict = parse_people(text)
from pprint import pprint
pprint(people_dict)


def parse_people_cell(cell):
    """
    Zwraca słownik:
    {
        'Bazowa nazwa osoby': {
            'wikidata': 'https://www.wikidata.org/wiki/Q....' lub None,
            'forms': [lista_form_użytych_w_tekście]
        },
        ...
    }
    """
    people = {}
    if not isinstance(cell, str) or not cell.strip():
        return people

    # Osoby rozdzielone są znakiem '|'
    chunks = [c.strip() for c in cell.split('|') if c.strip()]

    for chunk in chunks:
        base = None
        wikidata = None
        forms = []

        # Przypadek z identyfikatorem Wikidaty
        if '@' in chunk:
            base, rest = chunk.split('@', 1)
            base = base.strip()

            if '$' in rest:
                wikidata, forms_str = rest.split('$', 1)
                wikidata = wikidata.strip()
            else:
                wikidata = rest.strip()
                forms_str = ""
        else:
            # Przypadek bez Wikidaty (np. "Papalini$Papalini")
            if '$' in chunk:
                base, forms_str = chunk.split('$', 1)
            else:
                base, forms_str = chunk, ""
            base = base.strip()
            wikidata = None

        forms_str = forms_str.strip()

        # Użyte formy rozdzielone są ';'
        if forms_str:
            for part in forms_str.split(';'):
                f = part.strip()
                if not f:
                    continue
                # Na wszelki wypadek usuwamy początkowe '$' jeśli jest
                if f.startswith('$'):
                    f = f[1:].strip()
                if f:
                    forms.append(f)

        if base:
            people[base] = {
                "wikidata": wikidata if wikidata else None,
                "forms": forms
            }

    return people

# 3. Akumulacja osób z całej tabeli
people_wikidata = {}          # bazowa_nazwa -> wikidata (lub None)
people_forms = defaultdict(set)  # bazowa_nazwa -> zbiór form

for cell in df_test["Historical Figures Mentioned"]:
    cell_people = parse_people_cell(cell)
    for base, info in cell_people.items():
        wikidata = info["wikidata"]

        # jeśli osoba już istnieje, nie nadpisujemy istniejącej Wikidaty,
        # ale uzupełniamy ją, gdy była wcześniej None
        if base not in people_wikidata:
            people_wikidata[base] = wikidata
        else:
            if people_wikidata[base] is None and wikidata:
                people_wikidata[base] = wikidata

        # dokładamy wszystkie formy do zbioru (unikalne)
        for form in info["forms"]:
            people_forms[base].add(form)

# 4. Budowa kartoteki jako DataFrame
kartoteka = pd.DataFrame(
    [
        {
            "nazwa osoby": base,
            "wikidata": people_wikidata[base],
            "użyte formy": "; ".join(sorted(people_forms.get(base, [])))
        }
        for base in sorted(people_wikidata.keys())
    ]
)




































