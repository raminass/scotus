from utils import *
import subprocess
import sys
import re
import pandas as pd
from eyecite import find, clean
from IPython.display import display, HTML


# @title
def full_case(citation, text):
    text = text.replace(citation.matched_text(), "")
    if citation.metadata.year:
        pattern = r"\([^)]*{}\)".format(
            citation.metadata.year
        )  # Matches any word that ends with "year"
        text = re.sub(pattern, "", text)
    if citation.metadata.pin_cite:
        text = text.replace(citation.metadata.pin_cite, "")
    if citation.metadata.parenthetical:
        text = text.replace(f"({citation.metadata.parenthetical})", "")
    if citation.metadata.plaintiff:
        text = text.replace(
            f"{citation.metadata.plaintiff} v. {citation.metadata.defendant}", ""
        )
    publisher_date = " ".join(
        i for i in (citation.metadata.court, citation.metadata.year) if i
    )
    if publisher_date:
        text = text.replace(f"{publisher_date}", "")
    if citation.metadata.extra:
        text = text.replace(citation.metadata.extra, "")
    return text


def supra_case(citation, text):
    text = text.replace(citation.matched_text(), "")
    if citation.metadata.pin_cite:
        text = text.replace(citation.metadata.pin_cite, "")
    if citation.metadata.parenthetical:
        text = text.replace(f"({citation.metadata.parenthetical})", "")
    if citation.metadata.antecedent_guess:
        text = text.replace(citation.metadata.antecedent_guess, "")
    return text


def short_case(citation, text):
    text = text.replace(citation.matched_text(), "")
    if citation.metadata.parenthetical:
        text = text.replace(f"({citation.metadata.parenthetical})", "")
    if citation.metadata.year:
        pattern = r"\([^)]*{}\)".format(citation.metadata.year)
    if citation.metadata.antecedent_guess:
        text = text.replace(citation.metadata.antecedent_guess, "")
    return text


def id_case(citation, text):
    text = text.replace(citation.matched_text(), "")
    if citation.metadata.parenthetical:
        text = text.replace(f"({citation.metadata.parenthetical})", "")
    if citation.metadata.pin_cite:
        text = text.replace(citation.metadata.pin_cite, "")
    return text


def unknown_case(citation, text):
    text = text.replace(citation.matched_text(), "")
    if citation.metadata.parenthetical:
        text = text.replace(f"({citation.metadata.parenthetical})", "")
    return text


def full_law_case(citation, text):
    text = text.replace(citation.matched_text(), "")
    if citation.metadata.parenthetical:
        text = text.replace(f"({citation.metadata.parenthetical})", "")
    return text


def full_journal_case(citation, text):
    text = text.replace(citation.matched_text(), "")
    if citation.metadata.year:
        pattern = r"\([^)]*{}\)".format(
            citation.metadata.year
        )  # Matches any word that ends with "year"
        text = re.sub(pattern, "", text)
    if citation.metadata.pin_cite:
        text = text.replace(citation.metadata.pin_cite, "")
    if citation.metadata.parenthetical:
        text = text.replace(f"({citation.metadata.parenthetical})", "")
    return text


def all_commas(text: str) -> str:
    return re.sub(r"\,+", ",", text)


def all_dots(text: str) -> str:
    return re.sub(r"\.+", ".", text)


functions_dict = {
    "FullCaseCitation": full_case,
    "SupraCitation": supra_case,
    "ShortCaseCitation": short_case,
    "IdCitation": id_case,
    "UnknownCitation": unknown_case,
    "FullLawCitation": full_law_case,
    "FullJournalCitation": full_journal_case,
}


# @title
def remove_citations(input_text):
    # clean text
    plain_text = clean.clean_text(
        input_text, ["html", "inline_whitespace", "underscores"]
    )
    # remove citations
    found_citations = find.get_citations(plain_text)
    for citation in found_citations:
        plain_text = functions_dict[citation.__class__.__name__](citation, plain_text)
    # clean text
    plain_text = clean.clean_text(
        plain_text,
        ["inline_whitespace", "underscores", "all_whitespace", all_commas, all_dots],
    )
    plain_text = clean.clean_text(plain_text, ["inline_whitespace", "all_whitespace"])
    pattern = r"\*?\d*\s*I+\n"
    plain_text = re.sub(pattern, "", plain_text)
    pattern = r"\s[,.]"
    plain_text = re.sub(pattern, "", plain_text)
    return plain_text


def split_text(text):
    words = text.split()
    chunks = []
    for i in range(0, len(words), 420):
        chunks.append(" ".join(words[i : i + 430]))
    return chunks


# @title
def chunk_text_to_paragraphs(text):
    paragraphs = text.split("\n")  # Split by empty line

    # Remove leading and trailing whitespace from each paragraph
    paragraphs = [p.strip() for p in paragraphs]

    return paragraphs


# @title
def split_data(data, id2label, label2id):
    data_dict = {
        "author": [],
        "label": [],
        "type": [],
        "case_name": [],
        "text": [],
    }
    opinions_split = pd.DataFrame(data_dict)
    opinions_split["label"] = opinions_split["label"].astype(int)
    for index, row in data.iterrows():
        # chunks = chunk_text_to_paragraphs(row['text'])
        chunks = split_text(row["clean_text"])
        for chunk in chunks:
            if len(chunk) <= 1000:
                continue
            tmp = pd.DataFrame(
                {
                    "author": row["author"],
                    "label": [label2id[row["author"]]],
                    "type": row["type"],
                    "case_name": row["case_name"],
                    "text": [chunk],
                }
            )
            opinions_split = pd.concat([opinions_split, tmp])
    return opinions_split


def chunk_data(data):
    data_dict = {"text": []}
    opinions_split = pd.DataFrame(data_dict)
    chunks = split_text(data)
    for chunk in chunks:
        if len(chunk) < 1000:
            continue
        tmp = pd.DataFrame({"label": [200], "text": [chunk]})
        opinions_split = pd.concat([opinions_split, tmp])
    return opinions_split


def get_labels_maping(df):
    df = df[df.type != "per_curiam"]
    judges_list = df["author"].unique().tolist()
    # map judge name to index
    judge2idx = {judge: idx for idx, judge in enumerate(judges_list)}
    # judge2idx["per_curiam"] = 200
    # map index to judge name
    idx2judge = {idx: judge for idx, judge in enumerate(judges_list)}
    # idx2judge[200] = "per_curiam"
    return idx2judge, judge2idx


def find_case_by_name(df, name):
    names_list = df["case_name"].str.lower()
    return display(
        HTML(
            df[names_list.str.contains(name.lower())]
            .iloc[:, :-1]
            .to_html(render_links=True, escape=False)
        )
    )
