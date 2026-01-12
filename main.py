import collections
import datetime
from json.encoder import INFINITY
from math import inf
import math
from dateutil import relativedelta
import streamlit as st


import os
from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings, ChatOllama
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS 
from langchain_community.vectorstores.utils import DistanceStrategy


from pprint import pprint

# Context Enrichment / Summarization
OLLAMA_LANGUAGE_MODEL = "gemma3:270m"
LANGUAGE_MODEL_TEMPERATURE = 0.1
LANGUAGE_MODEL_REASONING = False

# Embedding
OLLAMA_EMBEDDING_MODEL = "embeddinggemma:300m"
TOKEN_CONTEXT_LENGTH = 1024
DOCUMENT_PREFIX = f"title: none | text: "
DOCUMENT_PREFIX_WITH_TITLE = lambda x : f"title: {x if x is not None else 'none'} | text: "
QUERY_PREFIX = "task: search result | query: "

embeddings = OllamaEmbeddings \
(
    model=OLLAMA_EMBEDDING_MODEL,
    num_ctx=TOKEN_CONTEXT_LENGTH,
    mirostat_eta = 0.1,
    mirostat_tau = 5.0,
    mirostat = 0,
    tfs_z=1.0
)

db_repo = FAISS.load_local("data/repo", embeddings, allow_dangerous_deserialization=True)
db_summary = FAISS.load_local("data/summary", embeddings, allow_dangerous_deserialization=True)


print(db_repo.index)
# State Variables
if "search_value" not in st.session_state:
    st.session_state['search_value'] = ""

if "summary_slider_value" not in st.session_state:
    st.session_state['summary_slider_value'] = 0.4

if "repo_slider_value" not in st.session_state:
    st.session_state['repo_slider_value'] = 0.4

if "star_slider_value" not in st.session_state:
    st.session_state['star_slider_value'] = 0.2

if "excluded_repos" not in st.session_state:
    st.session_state['excluded_repos'] = list()

# pprint(db_summary.distance_strategy)


print(st.session_state['excluded_repos'])

def update_sliders( changed_slider_key : str):

    other_keys : list = ['summary_slider_value', 'repo_slider_value', 'star_slider_value']

    other_keys.remove(changed_slider_key)

    new_val = st.session_state[changed_slider_key]
    remaining = 1.0 - new_val

    sum_others = sum(st.session_state[k] for k in other_keys)

    if sum_others > 0:

        for k in other_keys:

            st.session_state[k] = (st.session_state[k] / sum_others) * remaining

    else:

        for k in other_keys:
            st.session_state[k] = remaining / len(other_keys)


def exclude_repo(repo_origin : str):
    st.session_state['excluded_repos'].append(repo_origin)

def search():

    if st.session_state['search_value'].strip() != "":

        with results_container:

            st.markdown("##### Search results:", text_alignment="left")

            results = []
            # document schema
            # {
            #     document : Document()
            #     summary_idx: []
            #     summary_score: []
            #     repo_idx: []
            #     repo_score: []
            # }

            # pprint(list(f"{doc.metadata.get('repo_name')} ; {val} : {doc.page_content[:200]} " for (doc, val) in db_repo.similarity_search_with_score(QUERY_PREFIX + "python content management system", k=100)), width=1000)
            
            for (index, (doc, val)) in enumerate(db_summary.similarity_search_with_score(QUERY_PREFIX + st.session_state['search_value'] , k=100)):
                
                    if doc.metadata['repo_origin'] not in list(a["document"].metadata["repo_origin"] for a in results):
                        
                        results.append({ "document" : doc, "summary_idx" : [index],  "summary_score" : [val], "repo_idx" : [], "repo_score" : []})

                    else:

                        index_ = next((idx for (idx, d) in enumerate(results) if d["document"].metadata['repo_origin'] == doc.metadata['repo_origin']), None) 

                        if index_ is not None:

                            results[index_]["summary_idx"].append(index)
                            results[index_]["summary_score"].append(val.real)

            for (index, (doc, val)) in enumerate(db_repo.similarity_search_with_score(QUERY_PREFIX + st.session_state['search_value'] , k=100)):

                if doc.metadata['repo_origin'] not in list(a["document"].metadata["repo_origin"] for a in results):
                    results.append({ "document" : doc, "summary_idx" : [],  "summary_score" : [], "repo_idx" : [index],  "repo_score" : [val]})

                else:

                    index_ = next((idx for (idx, d) in enumerate(results) if d["document"].metadata['repo_origin'] == doc.metadata['repo_origin']), None)

                    if index_ is not None:

                        results[index_]["repo_idx"].append(index)
                        results[index_]["repo_score"].append(val.real)

            # {
            #     document : Document()
            #     summary_score: float (0 ; 1)
            #     repo_score : float (0 ; 1)
            #     star_score : flaot (0 ; 1)
            # }

            # pprint(results)

            min_summary = min(min(x["summary_score"] ) if x["summary_score"] else inf for x in results)
            max_summary = max(max(x["summary_score"] ) if x["summary_score"] else 0.0 for x in results)

            min_repo = min(min(x["repo_score"] ) if x["repo_score"] else inf for x in results)
            max_repo = max(max(x["repo_score"] ) if x["repo_score"] else 0.0 for x in results)

            min_stars = math.log( 1 + min(x["document"].metadata.get('repo_star_count')  for x in results))
            max_stars = math.log(1 + max(x["document"].metadata.get('repo_star_count') for x in results))

            print("-- summary --")

            print(min_summary)
            print(max_summary)

            print("-- repo --")

            print(min_repo)
            print(max_repo)

            print("-- stars --")

            print(min_stars)
            print(max_stars)

            print("-- results --")

            docs_with_scores = []

            for obj in results:

                doc_with_scores = \
                {
                    "document" : obj["document"],
                    "summary_score" : (max_summary - (min(obj["summary_score"]) if obj["summary_score"] else max_summary)) / (max_summary - min_summary),
                    "repo_score" : (max_repo - (min(obj["repo_score"]) if obj["repo_score"] else max_repo)) / (max_repo - min_repo),
                    "star_score" : 1 - (max_stars - math.log(1 + obj["document"].metadata.get('repo_star_count')) ) / (max_stars - min_stars)
                }

                doc_with_scores.update({"score" : (st.session_state['summary_slider_value'] * doc_with_scores["summary_score"]) + (st.session_state['repo_slider_value'] * doc_with_scores["repo_score"]) + (st.session_state['star_slider_value'] * doc_with_scores["star_score"])})

                # print(doc_with_scores["document"].metadata["repo_name"] , "-", doc_with_scores["score"])

                docs_with_scores.append(doc_with_scores)

            
            docs_with_scores = sorted(docs_with_scores, key = lambda item: item["score"], reverse = True)

            for (index, result) in enumerate(docs_with_scores):

                doc = result["document"]

                if doc.metadata['repo_origin'] not in st.session_state['excluded_repos']:

                    print(f"{result['document'].metadata['repo_name']} - {result['score']} \n\t( sum : {(st.session_state['summary_slider_value'] * result['summary_score']):.2f} ; repo : {(st.session_state['repo_slider_value'] * result['repo_score']):.2f} ; stars : {(st.session_state['star_slider_value'] * result['star_score']):.2f})")

                    r = relativedelta.relativedelta(datetime.datetime.now(datetime.timezone.utc), doc.metadata['repo_last_update_date'].replace(tzinfo=datetime.timezone.utc))

                    months_difference = (r.years * 12) + r.months

                    # print(months_difference)

                    with st.expander(f"**{doc.metadata['repo_origin']}**"):

                        with st.container(horizontal=True):

                            st.badge(f"**{doc.metadata['repo_star_count']}** stars" , icon=":material/star:", color="yellow")

                            st.badge(f"**{doc.metadata['repo_fork_count']}** forks" , icon=":material/arrow_split:", color="blue")

                            st.badge(f"Last update: **{'under 1 year' if months_difference < 12 else 'over 1 year'}**" , icon=":material/av_timer:", color='green' if months_difference < 12 else 'red')
                        
                        st.markdown(f"{doc.metadata['repo_description']}")

                        with st.container(horizontal=True, key=f"link_container_{index}"):\

                            st.link_button(label="Repository", url=f"https://github.com/{doc.metadata['repo_origin']}", icon=":material/commit:", type="tertiary")

                            st.button(label=f"{doc.metadata.get('repo_license_name')}", icon=":material/license:", type="tertiary" ,  key=f"license_button_{index}")

                            st.button(label="", icon=":material/block:", type="tertiary", help="Exclude this repo from any further searches.", key=f"block_button_{index}", on_click=exclude_repo , args=(doc.metadata['repo_origin'],))
        
        
 
    
title = st.markdown("""# Github Librarian""", text_alignment="center")

search_input = st.text_input(label= "Search Input",label_visibility="collapsed",placeholder = "What are you looking for?", value=st.session_state['search_value'].strip(), max_chars = 4000, key="search_value", on_change=search)

st.markdown("##### Weigths:", text_alignment="left")

summary_slider = st.slider("Summary Chunks", min_value=0.0, max_value=1.0, format="%0.2f", key='summary_slider_value', on_change=update_sliders, args=('summary_slider_value',))

repo_slider = st.slider("Repo Chunks", min_value=0.0, max_value=1.0, format="%0.2f", key = 'repo_slider_value', on_change=update_sliders, args=('repo_slider_value',))

star_count_slider = st.slider("Star Count", min_value=0.0, max_value=1.0,  format="%0.2f",key = 'star_slider_value', on_change=update_sliders, args=('star_slider_value',))

results_container = st.container(horizontal_alignment="center", border=False, vertical_alignment="center")

search()

