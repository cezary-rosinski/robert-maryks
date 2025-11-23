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
df_people = gsheet_to_df('1kxnFmIx1-DW2oIpb67l-uTSWcEzn-k9yykLJ1TE1TKQ', 'People')
df_places = gsheet_to_df('1M2gc-8cGZ8gh8TTnm4jl430bL4tccdri9nw-frUWYLQ', 'places')
df_events = gsheet_to_df('1M2gc-8cGZ8gh8TTnm4jl430bL4tccdri9nw-frUWYLQ', 'events')
df_genres = gsheet_to_df('1M2gc-8cGZ8gh8TTnm4jl430bL4tccdri9nw-frUWYLQ', 'genres')

#%%
people_names_ids = dict(zip(df_people['person_name'].to_list(), df_people['person_ID'].to_list()))

test = list(zip(df_test['letter_ID'], df_test['Historical Figures Mentioned'].to_list()))
result = []
for a,b in test:
    if pd.notna(b):
        b = [e.strip() for e in b.split('|')]
        for c in b:
            main_name = c.split('@')[0].strip()
            if '$' in c:
                used_name = [el.strip() for el in c.split('$')[-1].strip().split(';')]
            else: used_name = []
            if '$' in main_name:
                main_name = main_name.split('$')[0].strip()
            if main_name:
                person_id = people_names_ids.get(main_name)
                result.append((a,main_name,person_id,used_name))
            
used_forms = pd.DataFrame(result, columns=['letter_ID', 'person_name', 'person_ID', 'person_name_form'])

mentioned_in_letters = {}
for l,n,p,f in result:
    mentioned_in_letters.setdefault(l,[]).append(p)
mentioned_in_letters = {k:';'.join(v) for k,v in mentioned_in_letters.items()}
        
mentioned_in_letters_df = pd.DataFrame(mentioned_in_letters.items())

test = [ele for sup in [[re.split(f"[$]", el) for el in e.split('|')] for e in test] for ele in sup]
test = [[el.strip() for el in e] for e in test]
test = [[el.split('@') for el in e] for e in test]

test_df = pd.DataFrame(test)


#%%


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




































