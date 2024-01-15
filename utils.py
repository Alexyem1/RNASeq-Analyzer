# utils.py

import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from goatools import obo_parser
from goatools.semantic import TermCounts, resnik_sim

def load_data(file):
    """Load data from a file."""
    return pd.read_csv(file)

# Add spacing if this column is shorter
def add_spacer(num):
    for _ in range(num):
        st.write("")

#def filter_low_counts(data, cutoff):
#    """Filter out low count genes."""
#    return data.loc[:, (data > cutoff).all(axis=0)]

#def normalize_data(data):
#    """Normalize the data."""
#    return data / data.sum()

#def log_transform(data):
#    """Apply log transformation to the data."""
#    return np.log2(data + 1)

#def load_go_terms(go_obo_path):
#    """Load GO terms from the OBO file."""
#    return obo_parser.GODag(go_obo_path)





################################
####Scholar Scraper functions###
################################
import streamlit as st
import re
import requests
from bs4 import BeautifulSoup
import pandas as pd


# this function for the getting inforamtion of the web page
def get_paperinfo(paper_url, headers):

    #download the page
    response=requests.get(paper_url,headers=headers)

    # check successful response
    if response.status_code != 200:
        print('Status code:', response.status_code)
        raise Exception('Failed to fetch web page ')

    #parse using beautiful soup
    paper_doc = BeautifulSoup(response.text,'html.parser')
    for div in paper_doc.find_all("div", {'class':'gs_ggs gs_fl'}): 
        div.decompose()

    return paper_doc

# this function for the extracting information of the tags
def get_tags(doc):
    paper_tag = doc.select('[data-lid]')
    # cite_tag = doc.select('[title=Cite] + a')
    cite_tag = doc.find_all('div', {"class": "gs_fl"})
    link_tag = doc.find_all('h3',{"class" : "gs_rt"})
    author_tag = doc.find_all("div", {"class": "gs_a"})

    return paper_tag,cite_tag,link_tag,author_tag

    # it will return the title of the paper
def get_papertitle(paper_tag):
    
    paper_names = []
    
    for tag in paper_tag:
        paper_names.append(tag.select('h3')[0].get_text())

    return paper_names

# it will return the number of citation of the paper
def get_citecount(cite_tag):
    cite_count = []
    for i in cite_tag:
        cite = i.text
        tmp = re.findall('Cited by[ ]\d+', cite)
        if tmp:
            cite_count.append(tmp[0])
        else:
            cite_count.append(0)

    return cite_count

# function for the getting link information
def get_link(link_tag):

    links = []

    for i in range(len(link_tag)) :
        if link_tag[i].a:  
            links.append(link_tag[i].a['href']) 
        else:
            links.append(None)

    return links 

# function for the getting autho , year and publication information
def get_author_year_publi_info(authors_tag):
    years = []
    publication = []
    authors = []
    for i in range(len(authors_tag)):
        authortag_text = (authors_tag[i].text).split()
        # year = int(re.search(r'\d+', authors_tag[i].text).group())
        # years.append(year)
        
        input_text_year = " ".join(authors_tag[i].text.split()[-3:])
        datesearch = re.findall("(19\d{2}|20\d{2})", input_text_year)
        if len(datesearch) > 0:
            year = int(datesearch[len(datesearch)-1])
            years.append(year)
        else:
            year = 0
            years.append(year)
        publication.append(authortag_text[-1])
        author = authortag_text[0] + ' ' + re.sub(',','', authortag_text[1])
        authors.append(author)
    
    return years , publication, authors

def cite_number(text):
    if text != 0:
        result = text.split()[-1]
    else:
        result = str(text)
    return result

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

#Pubmed literature search
import streamlit as st
from Bio import Entrez
import pandas as pd
import time
from datetime import datetime, timedelta


def fetch_pubmed_abstracts(query, max_results=20):
    # Set your email here
    Entrez.email = "your.email@example.com"

    # Calculate date 10 years ago
    ten_years_ago = datetime.now() - timedelta(days=365*10)
    mindate = ten_years_ago.strftime('%Y/%m/%d')
    maxdate = datetime.now().strftime('%Y/%m/%d')
    search_results_old = Entrez.read(
        Entrez.esearch(
            db="pubmed", term=query, reldate=365, datetype="pdat", usehistory="y"
        )
    )


    search_results = Entrez.read(
        Entrez.esearch(
            db="pubmed",
            term=query,
            mindate=mindate,
            maxdate=maxdate,
            datetype="pdat",
            usehistory="y"
        )
    )


    count = int(search_results["Count"])
    abstracts = []

    for start in range(0, min(count, max_results), 10):
        end = min(count, start + 10)
        stream = Entrez.efetch(
            db="pubmed",
            rettype="medline",
            retmode="text",
            retstart=start,
            retmax=10,
            webenv=search_results["WebEnv"],
            query_key=search_results["QueryKey"],
        )


        
        data = stream.read()
        stream.close()
        abstracts.append(data)  # Store the fetched data in a list

        # Sleep for 20 seconds to respect request ethics
        time.sleep(20)

    return abstracts


from Bio import Medline
import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder
from pubmed_lookup import PubMedLookup, Publication


def display_results_in_aggrid(pubmed_results):
    # NCBI will contact user by email if excessive queries are detected
    email = ''

    # List to store parsed data
    parsed_results = []

    for result in pubmed_results:
        # Parse the MEDLINE formatted text
        records = Medline.parse(result.splitlines())
        for record in records:
            # Extract relevant information
            title = record.get("TI", "No title available")
            year = record.get("DP", "No date available").split()[0]
            authors = ', '.join(record.get("AU", ["No authors available"]))
            journal = record.get("JT", "No journal available")
            article_id = record.get("PMID", "")
            url = f"https://pubmed.ncbi.nlm.nih.gov/{article_id}/"

            # Append to the results list
            parsed_results.append({
                "Paper ID": article_id,
                "Paper Title": title,
                "Year": year,
                "Author": authors,
                "Publication Journal": journal,
                "Paper URL": url
            })

    # Convert to DataFrame
    df = pd.DataFrame(parsed_results)

    # Create AgGrid options
    grid_options = GridOptionsBuilder.from_dataframe(df)
    grid_options.configure_selection('multiple', use_checkbox=True)

    # Display the AgGrid table with checkboxes
    with st.expander("AgGrid Table"):
        grid_result = AgGrid(
            df,
            gridOptions=grid_options.build(),
            enable_enterprise_modules=False,
        )

    # Get selected rows
    selected_rows = grid_result['selected_rows']

    # Extract row indices from selected rows
    #selected_indices = [df.index[df['Paper Title'] == row['Paper Title']].tolist()[0] for row in selected_rows]


    # Display detailed information for selected rows
    #if selected_indices:
    #    st.subheader("Selected Data:")
    #    selected_data = df.iloc[selected_indices]
    #    st.write(selected_data)
    
    # Display the DataFrame using AgGrid
    #AgGrid(df)

    # Display detailed information for selected rows
    if selected_rows:
        st.subheader("Selected Data:")
        for row in selected_rows:
            #st.write(row)
            # Check if a publication is selected
            if 'Paper ID' in row:
                lookup = PubMedLookup(row["Paper ID"], email)
                publication = Publication(lookup)    # Use 'resolve_doi=False' to keep DOI URL
                st.write(
                    """
                    TITLE:{title}\n
                    AUTHORS:{authors}\n
                    JOURNAL:{journal}\n
                    YEAR:{year}\n
                    MONTH:{month}\n
                    DAY:{day}\n
                    URL:{url}\n
                    PUBMED:{pubmed}\n
                    CITATION:{citation}\n
                    MINICITATION:{mini_citation}\n
                    ABSTRACT:\n{abstract}\n
                    """
                    .format(**{
                        'title': publication.title,
                        'authors': publication.authors,
                        'journal': publication.journal,
                        'year': publication.year,
                        'month': publication.month,
                        'day': publication.day,
                        'url': publication.url,
                        'pubmed': publication.pubmed_url,
                        'citation': publication.cite(),
                        'mini_citation': publication.cite_mini(),
                        'abstract': repr(publication.abstract),
                    }))
                #st.subheader("Abstract:")
                #st.components.v1.html(f'<iframe src="{row["Paper URL"]}" width=800 height=600></iframe>', height=600)


import streamlit as st
import pandas as pd
import plotly.express as px

# Assuming you have 'fetch_pubmed_abstracts' and 'display_results_in_aggrid' functions defined

# Calculate the journal distribution
def calculate_journal_distribution(pubmed_results):
    journal_counts = {}
    for result in pubmed_results:
        records = Medline.parse(result.splitlines())
        for record in records:
            journal = record.get("JT", "No journal available")
            journal_counts[journal] = journal_counts.get(journal, 0) + 1
    return journal_counts







from pyvis.network import Network
from Bio import Medline
import streamlit as st
from stvis import pv_static

def create_authors_network(pubmed_results):
    if not pubmed_results:  # Check if pubmed_results is empty or None
        return None

    # Create a PyVis network
    nt = Network(height='500px', width='1000px', directed=False)

    authors_set = set()  # Use a set to store authors

    for result in pubmed_results:
        records = Medline.parse(result.splitlines())
        for record in records:
            authors = record.get("AU", [])
            if authors:
                for author in authors:
                    authors_set.add(author)  # Add author to the set

    # Add nodes for authors
    for author in authors_set:
        nt.add_node(author)

    # Add edges for co-authors
    for result in pubmed_results:
        records = Medline.parse(result.splitlines())
        authors = [record.get("AU", []) for record in records]
        for coauthors in authors:
            for i, author1 in enumerate(coauthors):
                for j, author2 in enumerate(coauthors):
                    if i != j:
                        nt.add_edge(author1, author2)

    return nt


#from streamlit_echarts import st_pyecharts
#from pyecharts import options as opts
#from pyecharts.charts import Bar
#import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
# Calculate publication counts for each author
def calculate_author_publication_counts(pubmed_results):
    author_counts = {}
    for result in pubmed_results:
        records = Medline.parse(result.splitlines())
        for record in records:
            authors = record.get("AU", [])
            for author in authors:
                author_counts[author] = author_counts.get(author, 0) + 1
    return author_counts

