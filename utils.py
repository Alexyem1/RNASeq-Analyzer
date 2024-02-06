# utils.py

# Data Handling and Transformation Functions:
    # read_file: For reading and loading data files.
    # filter_dataframe: For filtering DataFrame based on certain criteria.
    # split_frame: To split data into different frames for easier handling.
    # paginate_df: To manage large datasets by implementing pagination.
# Data Processing Functions:
    # load_volcano_data: Specific to loading data for volcano plots.
    # process_data: Generic data processing function.
# Plotting Functions:These functions are more about data visualization and less about UI/interaction logic.
    # create_bokeh_plot: For creating Bokeh plots.
    # create_corrmap: For generating correlation maps.
# Literature Handling Functions:These functions deal with processing and retrieving literature data, which is a backend operation.
    # fetch_literature: For fetching literature data.
    # filter_sentences: For filtering sentences in the literature.
    # get_full_text_deprecated: Older version of getting full text.
    # get_full_text: Current version of getting full text.

from __future__ import absolute_import
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import plotly.graph_objects as go

import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import streamlit as st
import re
import requests
from bs4 import BeautifulSoup
from Bio import Entrez
import time
from datetime import datetime, timedelta
from Bio import Medline
from st_aggrid import AgGrid, GridOptionsBuilder
from pubmed_lookup import PubMedLookup, Publication
from pyvis.network import Network
from stvis import pv_static
import base64
import io
from transformers import T5Tokenizer, T5ForConditionalGeneration
from xml.etree import ElementTree as ET
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
from bokeh.plotting import figure
#from bokeh.models import ColumnDataSource, HoverTool, CrosshairTool
from bokeh.models import ColumnDataSource, OpenURL, TapTool, HoverTool, CrosshairTool, WheelZoomTool
import dash_bio as dashbio

import streamlit as st
import streamlit.components.v1 as components

import jinja2
import pdfkit
import base64
from io import BytesIO

from PIL import Image
from datetime import datetime
import pytz
# Import Bokeh export functions
from bokeh.io.export import get_screenshot_as_png
from bs4 import BeautifulSoup





##############################################
# Data Handling and Transformation Functions:#
##############################################

#background image for the app
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover;
        

    }}
    </style>
    """,
    unsafe_allow_html=True
    )

@st.cache_data(show_spinner=True)
def read_file(uploaded_file):
    buffer = io.BytesIO(uploaded_file.getvalue())

    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(buffer)
    elif uploaded_file.name.endswith('.tsv'):
        return pd.read_csv(buffer, sep='\t')

# Add spacing if this column is shorter
def add_spacer(num):
    for _ in range(num):
        st.write("")

# Implementation of the filter_dataframe function to enable search filtering with multiselect and text input widgets.
def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox("Add filters")

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns, key="Filter Dataframe")
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            left.write("↳")
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    _min,
                    _max,
                    (_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].str.contains(user_text_input)]
    st.session_state["processed_df"] = df.reset_index(drop=True)

    return df.reset_index(drop=True)

#Pagination module
#@st.cache_data(show_spinner=False) <-- not sure wether I should use it here
def split_frame(input_df, rows):
    df = [input_df.loc[i : i + rows - 1, :] for i in range(0, len(input_df), rows)]
    return df

def paginate_df(dataset):
    top_menu = st.columns(3)
    with top_menu[0]:
        sort = st.radio("Sort Data", options=["Yes", "No"], horizontal=1, index=1)
    if sort == "Yes":
        with top_menu[1]:
            sort_field = st.selectbox("Sort By", options=dataset.columns)
        with top_menu[2]:
            sort_direction = st.radio(
                "Direction", options=["⬆️", "⬇️"], horizontal=True
            )
        dataset = dataset.sort_values(
            by=sort_field, ascending=sort_direction == "⬆️", ignore_index=True
        )
    pagination = st.container()

    bottom_menu = st.columns((4, 1, 1))
    with bottom_menu[2]:
        batch_size = st.selectbox("Page Size", options=[25, 50, 100])
    with bottom_menu[1]:
        total_pages = (
            int(len(dataset) / batch_size) if int(len(dataset) / batch_size) > 0 else 1
        )
        current_page = st.number_input(
            "Page", min_value=1, max_value=total_pages, step=1, key="Pagination"
        )
    with bottom_menu[0]:
        st.markdown(f"Page **{current_page}** of **{total_pages}** ")

    pages = split_frame(dataset, batch_size)
    pagination.dataframe(data=pages[current_page - 1], use_container_width=True)


#############################
# Data Processing Functions:#
#############################

# Cached function for loading data
@st.cache_data
def load_volcano_data(url):
    return pd.read_csv(url)

# Function to process data
def process_data(df, x_axis, y_axis, cutoff, record_dict):

    # Calculate fold change
    #df['fold_change'] = df.apply(lambda row: (row[x_axis] / row[y_axis]) if row[y_axis] != 0 else np.inf, axis=1)
    #df['fold_change'] = df.apply(
        #lambda row: (row[x_axis] / row[y_axis]) if row[y_axis] != 0 else float('inf') if row[x_axis] > 0 else float('-inf'),
        #axis=1
    #)
    def calculate_fold_change(row, x_axis, y_axis):
        if row[y_axis] == 0:
            # Handle the case when y_axis is zero
            if row[x_axis] > 0:
                return float('inf')
            elif row[x_axis] < 0:
                return float('-inf')
            else:
                return np.nan  # Handling the case when both x_axis and y_axis are zero
        else:
            # Handle the case when y_axis is not zero
            if row[x_axis] == 0:
                # If x_axis is zero, return a specific value (e.g., 0 or np.nan)
                return 0  # or np.nan, depending on how you wish to represent this scenario
            elif row[x_axis] >= row[y_axis]:
                return row[x_axis] / row[y_axis]
            else:
                return -1 / (row[x_axis] / row[y_axis])

    # Use the function in DataFrame.apply()
    df['fold_change'] = df.apply(lambda row: calculate_fold_change(row, x_axis, y_axis), axis=1)



    # Filter for significant genes based on cutoff
    significant_genes = df[(abs(df['fold_change']) > cutoff) & (df[x_axis] + df[y_axis] > 10)]

    if not df.empty:
            df.loc[:, 'seq'] = df['GeneID'].apply(
            lambda id: str(record_dict[id].seq) if id in record_dict else 'N/A'
        )

    if not significant_genes.empty:
        significant_genes.loc[:, 'x_values'] = significant_genes[x_axis]
        significant_genes.loc[:, 'y_values'] = significant_genes[y_axis]
        significant_genes.loc[:, 'seq'] = significant_genes['GeneID'].apply(
            lambda id: str(record_dict[id].seq) if id in record_dict else 'N/A'
        )
    return significant_genes, df


############################################################################################################
# Plotting Functions:These functions are more about data visualization and less about UI/interaction logic.#
############################################################################################################

# Create Bokeh plot
def create_bokeh_plot(df,significant_genes, x_axis, y_axis, gene_annotation):
    df = df.copy()
    df['x_values'] = df[x_axis]
    df['y_values'] = df[y_axis]
    df['color'] = 'blue'


    significant_genes['x_values'] = significant_genes[x_axis]
    significant_genes['y_values'] = significant_genes[y_axis]
    significant_genes['color'] = 'red'
    
    if gene_annotation:
        significant_genes.loc[significant_genes['Annotation'].str.contains(gene_annotation, case=False, na=False), 'color'] = 'yellow'
        df.loc[df['Annotation'].str.contains(gene_annotation, case=False, na=False), 'color'] = 'yellow'

    source = ColumnDataSource(significant_genes)
    source_df = ColumnDataSource(df)
    
    p = figure(title=f"{x_axis} vs {y_axis}", tools="pan,box_zoom,wheel_zoom,reset,save,tap")

 
    #p.circle(x_axis, y_axis,source=df, color="blue", alpha=1.0, muted_alpha=0.1, legend='all genes', size=7, 
             #line_color="black"
    #         )
    p.circle(x_axis, y_axis,source=source_df, color="color", alpha=1.0, muted_alpha=0.1, legend='all genes', size=7, line_color=None
            #line_color="black"
            )

    p.toolbar.active_scroll = p.select_one(WheelZoomTool)
    p.xaxis.axis_label = x_axis
    p.yaxis.axis_label = y_axis
    p.yaxis.axis_label_text_font_size = '14pt'
    p.xaxis.axis_label_text_font_size = '14pt'
    p.yaxis.major_label_text_font_size = '12pt'
    p.xaxis.major_label_text_font_size = '12pt'
    p.title.text_font_size = '16pt'
    p.circle('x_values', 'y_values', source=source, size=7, color='color', legend='sig. genes', line_color=None)




    # Add tools to the plot
    url = "http://papers.genomics.lbl.gov/cgi-bin/litSearch.cgi?query=@seq&Search=Search"
    taptool = p.select(type=TapTool)
    taptool.callback = OpenURL(url=url)
    hover = HoverTool(tooltips=[("GeneID", "@GeneID"), ("Fold Change", "@fold_change"), ("Annotation", "@Annotation")])
    p.add_tools(hover)
    p.add_tools(CrosshairTool())

    return p

# Function to create the correlation Clustergram
def create_corrmap(df, cols, colorscale, correlation_method):
    # Ensure DataFrame is numeric and compute correlation matrix
    df_numeric = df[cols].select_dtypes(include=[np.number]) 
    #df_corr = df_numeric.corr() # Compute correlation matrix
    df_corr = df_numeric.corr(method=correlation_method)
    #st.dataframe(df_corr.values)

    # Simplified Clustergram call
    try:
        fig = dashbio.Clustergram(
            data=df_corr.values,
            column_labels=df_corr.columns.tolist(),
            row_labels=df_corr.index.tolist(),
            color_map=colorscale,
            center_values=False,
        
        )

        # Layout Enhancements
        fig.update_layout(
            title_text='Correlation Matrix Clustergram',
            title_x=0.5,
            margin=dict(l=70, r=70, t=70, b=70),
            height=400,
            width=600
        )
    except ValueError as e:
        # Print error for debugging
        print("Error creating Clustergram:", e)
        print("Data shape:", df_corr.values.shape)
        print("Column labels:", df_corr.columns.tolist())
        print("Row labels:", df_corr.index.tolist())
        raise

    except Exception as e:
        st.error(f"Error creating clustergram: {e}")
        return None



    return fig



def create_corrmap_deprecated(df, cols, colorscale):
    # Ensure DataFrame is numeric and compute correlation matrix
    df_numeric = df[cols].select_dtypes(include=[np.number])
    df_corr = df_numeric.corr()  # Compute correlation matrix

        # Initialize session state variables
    if 'cluster_method' not in st.session_state:
        st.session_state.cluster_method = "all"
    if 'row_dist_method' not in st.session_state:
        st.session_state.row_dist_method = "euclidean"
    if 'col_dist_method' not in st.session_state:
        st.session_state.col_dist_method = "euclidean"
    if 'linkage_method' not in st.session_state:
        st.session_state.linkage_method = "complete"

    def update_cluster_method():
        st.session_state.cluster_method = cluster_method

    def update_row_dist_method():
        st.session_state.row_dist_method = row_dist_method

    def update_col_dist_method():
        st.session_state.col_dist_method = col_dist_method

    def update_linkage_method():
        st.session_state.linkage_method = linkage_method

    # Add selectors with session state in the sidebar
    with st.sidebar:
        cluster_method = st.selectbox("Select Cluster Method", ["all", "row", "column"],
                                    index=0, on_change=update_cluster_method)
        row_dist_method = st.selectbox("Select Row Distance Metric", ["euclidean", "minkowski", "cityblock"],
                                    index=0, on_change=update_row_dist_method)
        col_dist_method = st.selectbox("Select Column Distance Metric", ["euclidean", "minkowski", "cityblock"],
                                    index=0, on_change=update_col_dist_method)
        linkage_method = st.selectbox("Select Linkage Method", ["complete", "single", "average", "ward"],
                                    index=0, on_change=update_linkage_method)



    # Create the Clustergram
    try:
        fig = dashbio.Clustergram(
            data=df_corr.values,
            column_labels=df_corr.columns.tolist(),
            row_labels=df_corr.index.tolist(),
            color_map=colorscale,
            cluster=st.session_state.cluster_method,
            row_dist=st.session_state.row_dist_method,
            col_dist=st.session_state.col_dist_method,
            link_method=st.session_state.linkage_method
        )

        # Layout Enhancements
        fig.update_layout(
            title_text='Correlation Matrix Clustergram',
            title_x=0.5,
            margin=dict(l=40, r=40, t=40, b=40),
            height=400,
            width=600
        )
    except ValueError as e:
        print("Error creating Clustergram:", e)
        raise
    except Exception as e:
        st.error(f"Error creating clustergram: {e}")
        return None

    return fig



###################################################################################################################################
# Literature Handling Functions:These functions deal with processing and retrieving literature data, which is a backend operation.#
###################################################################################################################################
@st.cache_data
def fetch_literature(gene_name):
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query={gene_name}&format=json&resultType=core"

    progress_bar = st.progress(0)
    status_text = st.empty()

    summaries = []
    response = requests.get(url)
    if response.status_code == 200:
        articles = response.json().get('resultList', {}).get('result', [])
        #print(articles)

        # Initialize T5 model for summarization
        model_name = "t5-small"
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)

        for i, article in enumerate(articles[:3]):  # Limit to first 3 articles
            #print(article.get('pmcid'))
            progress_bar.progress((i+1)/3)
            status_text.text(f"Processing article {i+1}/3")

            try:
                article_ID = article.get('pmcid')
                #print(article_ID)
                #full_text = get_full_text(article_ID)
                full_text = article.get("abstractText")
                #print(full_text)
                if full_text:
                    # Ensure the gene name is included in the summary
                    relevant_sentences = filter_sentences(full_text, gene_name)
                    relevant_text = " ".join(relevant_sentences)
                    # Summarize the filtered text
                    inputs = tokenizer.encode("summarize: " + relevant_text, return_tensors="pt", max_length=512, truncation=True)
                    summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
                    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                    summaries.append((article.get('title', 'No Title'), summary))
                else:
                    summaries.append((article.get('title', 'No Title'), "Full text not available."))
            except Exception as e:
                summaries.append((article.get('title', 'No Title'), f"Error processing full text: {e}"))

        progress_bar.empty()
        status_text.empty()

    else:
        st.error("Failed to fetch data from Europe PMC")

    return summaries

def filter_sentences(text, gene_name):
    sentences = text.split('.')
    return [sentence.strip() + '.' for sentence in sentences if gene_name.lower() in sentence.lower()]

def get_full_text1(article_id):
    # Function to retrieve the full text of an article
    full_text_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{article_id}/fullTextXML"
    response = requests.get(full_text_url)
    if response.status_code == 200:
        return response.text  # Or parse XML to extract relevant sections
    return None

def get_full_text(article_id):
    full_text_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{article_id}/fullTextXML"
    #print(full_text_url)

    try:
        response = requests.get(full_text_url)
        if response.status_code == 200:
            # Parse the XML response
            tree = ET.fromstring(response.content)
            # Extract the full text from the XML
            # Note: This depends on the structure of the XML response
            # You may need to adjust the following line to match the actual XML structure
            full_text = tree.find('.//fullText').text
            #print(full_text)
            return full_text
        else:
            print(f"Failed to fetch full text for article ID {article_id}: HTTP {response.status_code}")
            return None
    except Exception as e:
        print(f"Error occurred while fetching full text for article ID {article_id}: {e}")
        return None


################################
####Scholar Scraper functions###
################################

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
@st.cache_data
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

#@st.cache_data(experimental_allow_widgets=True) 
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
            if 'Paper ID' in row:
                lookup = PubMedLookup(row["Paper ID"], email)
                publication = Publication(lookup)  # Use 'resolve_doi=False' to keep DOI URL

                soup = BeautifulSoup(publication.abstract, "html.parser")
                plain_text = soup.get_text()
                
                # Define CSS styles
                styles = """
                <style>
                    .publication-info {
                        font-family: Arial, sans-serif;
                        margin-bottom: 20px;
                    }
                    .publication-title {
                        font-weight: bold;
                        font-size: 20px;
                        color: #d66e13;
                    }
                    .publication-header {
                        font-weight: bold;
                    }
                </style>
                """

                # Structured publication information
                publication_info = f"""
                {styles}
                <div class="publication-info">
                    <div class="publication-title">{publication.title}</div>
                    <div><span class="publication-header">Authors:</span> {publication.authors}</div>
                    <div><span class="publication-header">Journal:</span> {publication.journal}</div>
                    <div><span class="publication-header">Published:</span> {publication.year}, {publication.month} {publication.day}</div>
                    <div><span class="publication-header">URL:</span> <a href="{publication.url}" target="_blank">{publication.url}</a></div>
                    <div><span class="publication-header">PubMed:</span> <a href="{publication.pubmed_url}" target="_blank">{publication.pubmed_url}</a></div>
                    <div><span class="publication-header">Citation:</span> {publication.cite()}</div>
                    <div><span class="publication-header">Mini-Citation:</span> {publication.cite_mini()}</div>
                </div>
                """

                # Use markdown to render the styled publication information
                st.markdown(publication_info, unsafe_allow_html=True)

                # Display abstract with an expander
                with st.expander("**Abstract:**"):
                    st.write(plain_text)


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

@st.cache_resource
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

# Function to create the Plotly Volcano plot
@st.cache_resource
def create_plotly_volcano_plot(df, sel_col_P, sel_col_FC, sel_col_ann, sel_col_id, gene_annotation, p_value_filter, fold_change_filter):
    # Data preparation
    df['color'] = 'blue'  # Default color for all genes
    df['-log10(p-value)'] = -np.log10(df[sel_col_P])

    # Coloring significant genes
    significant_filter = (df[sel_col_P] <= p_value_filter) & (abs(df[sel_col_FC]) >= fold_change_filter)
    df.loc[significant_filter, 'color'] = 'red'

    # Coloring genes matching the annotation
    if gene_annotation:
        annotation_filter = df[sel_col_ann].str.contains(gene_annotation, case=False, na=False)
        df.loc[annotation_filter, 'color'] = 'yellow'

    # Create Plotly figure
    fig = go.Figure()


    # Use go.Scatter instead of go.Scattergl
    fig.add_trace(go.Scattergl(
        x=df[sel_col_FC], 
        y=df['-log10(p-value)'],
        mode='markers',
        customdata=df[sel_col_id],  # Add geneID as custom data
        marker=dict(color=df['color'], size=7),
        hovertemplate="<b>%{customdata}</b><br>%{text}<br>Fold Change: %{x}<br>-log10(p-value): %{y}<extra></extra>",  # Custom hover template
        text=df[sel_col_ann]  # Data shown on hover
    ))

    # Update layout
    fig.update_layout(
        title='Volcano Plot',
        xaxis_title='Fold Change',
        yaxis_title='-log10(p-value)',
        hovermode='closest',
    )

        # Add annotations for a custom legend
    annotations = [
        dict(xref='paper', yref='paper', x=0.95, y=0.95, xanchor='left', yanchor='top',
             text='Default Genes', showarrow=False, font=dict(size=12, color='blue')),
        dict(xref='paper', yref='paper', x=0.95, y=0.85, xanchor='left', yanchor='top',
             text='Significant Genes', showarrow=False, font=dict(size=12, color='red')),
        dict(xref='paper', yref='paper', x=0.95, y=0.75, xanchor='left', yanchor='top',
             text='Annotation', showarrow=False, font=dict(size=12, color='yellow')),
        # Add more annotations as needed
    ]

    fig.update_layout(annotations=annotations)

    #fig.update_layout({"uirevision": "foo"}, overwrite=True) 
    fig.update_layout(uirevision=True)


    return fig

# Function to convert plots for Report generation
def plot_to_bytes(fig, graph_module="pyplot", format="png"):
    buf = BytesIO()
    if graph_module == "pyplot":
        fig.savefig(buf, format = format, bbox_inches="tight", dpi=300)
    elif graph_module == 'plotly':
        fig.write_image(file = buf, format = format, scale=3)
    elif graph_module == 'bokeh':
        # Bokeh handling
        get_screenshot_as_png(fig, driver=None, timeout=5, resources="cdn").save(buf, format=format)

    
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return data
