# main_app.py

#UI and Interaction Functions:
    # show_file_upload: Manages the UI for file upload.
    # show_missing_data: Displays missing data information in the UI.
    # show_data_preprocessing: Handles the UI for data preprocessing.
    # show_volcano_plot: Specifically for displaying the volcano plot in the UI.
    # show_literature_search, show_googlescholar_results, show_europepmc_results, show_pubmed_results: Manage the UI for literature search and results display.
    # generate_report_ui, feedback_ui: Handle UI elements for report generation and feedback.
# Core Functionalities:
    # main: The main function to run the app, which orchestrates the flow and interactions.

# Import necessary modules
import streamlit as st
from utils import add_spacer, fetch_pubmed_abstracts, display_results_in_aggrid, calculate_journal_distribution, calculate_author_publication_counts,create_authors_network
from utils import read_file, filter_dataframe, paginate_df, load_volcano_data, process_data, create_bokeh_plot, create_corrmap
import tempfile
from st_aggrid import AgGrid
import pandas as pd
from faker import Faker
import random
import io
from stvis import pv_static
import plotly.graph_objects as go
import numpy as np
from scipy.cluster import hierarchy as sch
from scipy.spatial.distance import pdist
#Literature search module, modified version of the code from https://github.com/nainiayoub/scholar-scrap:
import requests
import urllib
from bs4 import BeautifulSoup
import re
import time
from time import sleep
from utils import get_paperinfo, get_tags, get_papertitle, get_citecount, get_link, get_author_year_publi_info, cite_number, convert_df
from transformers import T5Tokenizer, T5ForConditionalGeneration
from xml.etree import ElementTree as ET
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
import missingno as msno
import matplotlib.pyplot as plt
import plotly.express as px
# ... other imports ...
import dash_bio as dashbio
from streamlit_plotly_events import plotly_events
from Bio import SeqIO
import os


# Define UI functions for each section of the app
    
def show_file_upload():
    uploaded_file = st.file_uploader("Upload your CSV/TSV file", type=['csv', 'tsv'], key="Data Upload")

    if uploaded_file is not None:
        df = read_file(uploaded_file)

        # Store the DataFrame in session state
        st.session_state['uploaded_data'] = df
        # Display the DataFrame using AgGrid
        #AgGrid(df)
        #st.dataframe(df)
        n, m = df.shape
        st.write("Data preview:")
        st.write(f'<p style="font-size:100%">Dataset contains {n} rows and {m} columns.</p>', unsafe_allow_html=True)
        st.write(df.head())
        st.success("File uploaded and displayed successfully!")
    
    # Check if there is a file stored in session state
    elif 'uploaded_data' in st.session_state:
        # Recreate the file from the stored data
        uploaded_file = st.session_state['uploaded_data']
        #AgGrid(uploaded_file)
        #st.dataframe(uploaded_file)
        st.write(uploaded_file.head())
        st.success("File uploaded and displayed successfully!")


def show_missing_data():
    if 'uploaded_data' in st.session_state and not st.session_state['uploaded_data'].empty:
        # Use the uploaded data from session state
        df = st.session_state['uploaded_data']

        # Create a figure for the missing data visualization
        #fig, ax = plt.subplots()
        fig, ax = plt.subplots(figsize=(6, 4))  # Adjust the figure size as needed
        #msno.bar(df, ax=ax)
        msno.matrix(df, ax=ax)

        # Display the figure in Streamlit
        st.pyplot(fig)


def show_data_preprocessing():
    """Displays the UI elements for data preprocessing using filter_dataframe."""
    st.subheader("Data Preprocessing")


    if 'uploaded_data' in st.session_state and not st.session_state['uploaded_data'].empty:
        # Use the uploaded data from session state
        df = st.session_state['uploaded_data']
        # Apply the filter_dataframe function to allow users to filter the data
        filtered_df = filter_dataframe(df)  # Assuming filter_dataframe is your custom function
         # Display the filtered dataframe without pagination
        #st.dataframe(filtered_df)
        # add tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Uploaded Data","Data Info", "Numeric Features", "Categorical Features"])
        with tab1:
            paginate_df(filtered_df)
        with tab2: #implementation is a modified version of the code from: https://pjoshi15.com/exploratory-data-analysis-app-streamlit/
            # extract meta-data from the uploaded dataset
            col1, col2, col3 = st.columns(3)
            with col1:
                #st.header("Meta-data")
                #st.markdown("#### Meta-Data")
                st.markdown("<h4 style='text-align: center'>Meta-Data</h4>", unsafe_allow_html=True)
                #st.write(df.style.set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]}, {'selector': 'td', 'props': [('text-align', 'center')]}]), unsafe_allow_html=True)
                row_count = df.shape[0]
                column_count = df.shape[1]
                
                # Use the duplicated() function to identify duplicate rows
                duplicates = df[df.duplicated()]
                duplicate_row_count =  duplicates.shape[0]
            
                missing_value_row_count = df[df.isna().any(axis=1)].shape[0]
            
                table_markdown = f"""
                | Description | Value | 
                |---|---|
                | Number of Rows | {row_count} |
                | Number of Columns | {column_count} |
                | Number of Duplicated Rows | {duplicate_row_count} |
                | Number of Rows with Missing Values | {missing_value_row_count} |
                """
                st.markdown(table_markdown)
                #show_missing_data()

            with col2:
                #st.header("Columns Type")
                #st.markdown("#### Columns Type")
                #st.markdown("<h4 style='text-align: center'>Columns Type</h4>", unsafe_allow_html=True)
    
                # get feature names
                #columns = list(df.columns)
                
                # create dataframe
                #column_info_table = pd.DataFrame({
                #    "column": columns,
                #    "data_type": df.dtypes.tolist()
                #})
                    
                # display pandas dataframe as a table
                #st.dataframe(column_info_table, hide_index=True)

                st.markdown("<h4 style='text-align: center'>Columns Type</h4>", unsafe_allow_html=True)
                # Create a DataFrame for Columns Type
                columns_type_df = pd.DataFrame(df.dtypes, columns=['Data Type']).reset_index()
                columns_type_df.rename(columns={'index': 'Column'}, inplace=True)
                # Convert DataFrame to HTML and use HTML to center the table
                st.markdown(columns_type_df.to_html(index=False, classes='center-table'), unsafe_allow_html=True)
                st.markdown("<style> .center-table { margin-left: auto; margin-right: auto; } </style>", unsafe_allow_html=True)
            
            with col3:
                #st.header("Missing Data")
                #st.markdown("#### Missing Data")
                st.markdown("<h4 style='text-align: center'>Missing Data</h4>", unsafe_allow_html=True)
                show_missing_data()

        with tab3:
            # find numeric features  in the dataframe
            numeric_cols = df.select_dtypes(include='number').columns.tolist()
 
            # add selection-box widget
            selected_num_col = st.selectbox("Which numeric column do you want to explore?", numeric_cols)

            # Create two columns for Statistics table and Histogram
            col1, col2 = st.columns(2)

            with col1:
                #st.header(f"{selected_num_col} - Statistics")
                st.markdown(f"<h4 style='text-align: center'>{selected_num_col} - Statistics</h4>", unsafe_allow_html=True)

                # Calculate and display statistics for the selected numeric column
                col_info = {}
                col_info["Number of Unique Values"] = len(df[selected_num_col].unique())
                col_info["Number of Rows with Missing Values"] = df[selected_num_col].isnull().sum()
                col_info["Number of Rows with 0"] = df[selected_num_col].eq(0).sum()
                col_info["Number of Rows with Negative Values"] = df[selected_num_col].lt(0).sum()
                col_info["Average Value"] = df[selected_num_col].mean()
                col_info["Standard Deviation Value"] = df[selected_num_col].std()
                col_info["Minimum Value"] = df[selected_num_col].min()
                col_info["Maximum Value"] = df[selected_num_col].max()
                col_info["Median Value"] = df[selected_num_col].median()
                
                info_df = pd.DataFrame(list(col_info.items()), columns=['Description', 'Value'])
                # Set the DataFrame index to the 'Description' column
                info_df.set_index('Description', inplace=True)
                # display dataframe as a markdown table and adjust height as needed
                #st.dataframe(info_df, height=500)
                #st.dataframe(info_df)
                # Additional columns for centering the DataFrame
                left, center, right = st.columns([1, 3.5, 1])
                with center:
                    st.dataframe(info_df)

                #info_df = pd.DataFrame(list(col_info.items()), columns=['Description', 'Value'])
                # Convert DataFrame to HTML and use HTML to center the table
                #st.markdown(info_df.to_html(index=False, classes='center-table'), unsafe_allow_html=True)
                #st.markdown("<style> .center-table { margin-left: auto; margin-right: auto; } </style>", unsafe_allow_html=True)




            with col2:
                #st.header("Histogram")
                st.markdown(f"<h4 style='text-align: center'>Histogram of {selected_num_col}</h4>", unsafe_allow_html=True)
                # Plot and display histogram for the selected numeric column
                #fig = px.histogram(df, x=selected_num_col)
                #fig.update_layout(height=400)  # Adjust height to align with the table
                #st.plotly_chart(fig, use_container_width=True)


                # Enhanced histogram
                fig = px.histogram(df, x=selected_num_col, marginal='box',  # Add a boxplot
                                   color_discrete_sequence=['indianred'],  # Color of the histogram
                                   hover_data=[selected_num_col],  # Data to show on hover
                                   labels={selected_num_col: 'Value'},  # Axis labeling
                                   template='plotly_dark')  # Plotly theme
                fig.update_layout(height=400, 
                                  #title_text=f'Histogram of {selected_num_col}', 
                                  #title_x=0.5,  # Center title
                                  bargap=0.2)  # Gap between bars
                # Make x-axis and y-axis titles and numbers bold
                #fig.update_xaxes(title_font=dict(size=12, color='white', family='Arial, sans-serif', weight='bold'),
                #                 tickfont=dict(family='Arial, sans-serif', color='white', weight='bold'))
                #fig.update_yaxes(title_font=dict(size=12, color='white', family='Arial, sans-serif', weight='bold'),
                #                 tickfont=dict(family='Arial, sans-serif', color='white', weight='bold'))
                fig.update_xaxes(title_font=dict(size=18, family='Courier', color='crimson'),ticks="inside",tickwidth=2, tickcolor='crimson')
                fig.update_yaxes(title_font=dict(size=18, family='Courier', color='crimson'),tickwidth=2, tickcolor='crimson')

                #st.plotly_chart(fig, use_container_width=True)
                # Additional columns for centering the histogram
                left, center, right = st.columns([0.5, 5, 1])
                with center:
                    st.plotly_chart(fig, use_container_width=True)

        with tab4:
            # find categorical columns in the dataframe
            cat_cols = df.select_dtypes(include='object')
            cat_cols_names = cat_cols.columns.tolist()
        
            # add select widget
            selected_cat_col = st.selectbox("Which text column do you want to explore?", cat_cols_names)
        
            #st.header(f"{selected_cat_col}")
            st.markdown(f"<h4 style='text-align: center'>Description of the categorical variable: {selected_cat_col}</h4>", unsafe_allow_html=True)
            
            # add categorical column stats
            cat_col_info = {}
            cat_col_info["Number of Unique Values"] = len(df[selected_cat_col].unique())
            cat_col_info["Number of Rows with Missing Values"] = df[selected_cat_col].isnull().sum()
            cat_col_info["Number of Empty Rows"] = df[selected_cat_col].eq("").sum()
            cat_col_info["Number of Rows with Only Whitespace"] = len(df[selected_cat_col][df[selected_cat_col].str.isspace()])
            cat_col_info["Number of Rows with Only Lowercases"] = len(df[selected_cat_col][df[selected_cat_col].str.islower()])
            cat_col_info["Number of Rows with Only Uppercases"] = len(df[selected_cat_col][df[selected_cat_col].str.isupper()])
            cat_col_info["Number of Rows with Only Alphabet"] = len(df[selected_cat_col][df[selected_cat_col].str.isalpha()])
            cat_col_info["Number of Rows with Only Digits"] = len(df[selected_cat_col][df[selected_cat_col].str.isdigit()])
            cat_col_info["Mode Value"] = df[selected_cat_col].mode()[0]
        
            cat_info_df = pd.DataFrame(list(cat_col_info.items()), columns=['Description', 'Value'])
            # Set the DataFrame index to the 'Description' column
            cat_info_df.set_index('Description', inplace=True)

            # Additional columns for centering the histogram
            left, center, right = st.columns([3.5, 5, 1])
            with center:
                st.dataframe(cat_info_df)

    else:
        st.write("Please upload a file in the Data Upload section.")


def show_volcano_plot():
    #st.title('Enhanced Volcano Plot Visualization with Streamlit and Dash')

    tab1, tab2, tab3 = st.tabs(["Volcano Plot", "Scatter Plot", "Clustergram"])

    with tab1:
        st.subheader("Volcano Plot")
        st.write("Explore different thresholds and colors to visualize the data:")

        # Load data
        df = load_volcano_data('https://git.io/volcano_data1.csv') # to do

        # Container for the Volcano Plot and table
        col1, col2, col3 = st.columns([0.2,0.2,0.2])
        with col1.container(border=None):
            #form = st.form(key='Volcano Plot')
            #st.write("Explore different thresholds and colors to visualize the data:")
            #st.subheader('P-value Threshold')
            #st.markdown('##### **P-value Threshold**', unsafe_allow_html=True)
            p_value_threshold = st.slider('P-value Threshold',label_visibility='visible', min_value=0.0, max_value=1.0, value=0.05, step=0.01)
        with col2.container(border=None):
            #st.subheader('Fold Change Threshold')
            #st.markdown('##### **Fold Change Threshold**', unsafe_allow_html=True)
            # Add a slider for fold change filtering
            fold_change_threshold = st.slider('Fold Change Threshold',label_visibility='visible', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        with col3.container(border=None):
            #st.subheader('Select Highlight Color')
            #st.markdown('##### **Select Highlight Color**', unsafe_allow_html=True)
            # Add color picker for customizing plot colors
            color = st.color_picker('Select Highlight Color', '#FFA500', label_visibility='visible',)  # Default is orange
        
        # Update filtering
        filtered_df = df[(df['P'] <= p_value_threshold) & (df['EFFECTSIZE'].abs() >= fold_change_threshold)]

        # Reset index after filtering
        filtered_df = filtered_df.reset_index(drop=True)

        st.markdown("---")

        col4, col5 = st.columns([0.5,0.5])
        with col4.container(border=True):
            # Check if filtered_df is not empty
            if len(filtered_df) > 0:
                # Redraw the Volcano Plot with the filtered data and customizations
                fig = dashbio.VolcanoPlot(
                    dataframe=filtered_df,
                    effect_size='EFFECTSIZE',  # Replace with your actual column name for fold change
                    p='P',  # Replace with your actual column name for P-value
                    gene='GENE',  # Replace with your actual column name for Gene
                    highlight_color=color,
                    point_size=9
                )

                # Edit Layout with customizations
                fig.update_layout({'width': 550, 'height': 500, 'showlegend': True, 'hovermode': 'closest'})


                # Capture selected data
                selected_data = plotly_events(fig, click_event=True, select_event=True, key="pe_selected")

                # Display selected data
                if selected_data:
                    selected_indices = [point["pointIndex"] for point in selected_data]
                    #st.write("Selected Data Indices:", selected_indices)
                    #filtered_selected_df = filtered_df.iloc[selected_indices]
                    #st.table(filtered_selected_df)

                    # Filter and display only the GENE column for these points
                    selected_genes = filtered_df.iloc[selected_indices]['GENE']
                    #st.write("Selected Genes:")
                    #st.table(selected_genes)

                    # Display the genes as a list
                    #st.write("Selected Genes:")
                    #for gene in selected_genes:
                    #    st.write(gene)

                    st.write("Selected Genes:")
                    for gene in selected_genes:
                        # Create a clickable link for each gene
                        if st.button(gene):
                            st.session_state['search_term'] = gene
                            # This line can optionally be used to scroll to the literature search section
                            # st.experimental_rerun()
                else:
                    st.write("No points selected.")


                # Streamlit app

                #st.write("Volcano Plot with Interactive Filters")
                #st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("No data to display for the selected thresholds.")
            add_spacer(4)

        with col5.container(border=None):
            #st.table(filtered_df)
            # Display paginated table of the filtered DataFrame
            paginate_df(filtered_df)

        
    with tab2:
        #Pagination module
        #@st.cache_data(show_spinner=False) <-- not sure wether I should use it here
        def split_frame(input_df, rows):
            df = [input_df.loc[i : i + rows - 1, :] for i in range(0, len(input_df), rows)]
            return df

        def paginate_filtered_data(dataset):
            top_menu = st.columns(3)
            with top_menu[0]:
                #fig = go.Figure(go.Indicator(
                #mode = "gauge+number",
                #value = 270,
                #domain = {'x': [0, 1], 'y': [0, 1]},
                #title = {'text': "Speed"}
                #))
                #fig.update_layout(
                #    width=200, height=400  # Added parameter
                #    )
                #st.plotly_chart(fig)
                # Create three columns with equal space
                st.metric(label="Gene count", value=len(filtered_data))
                #st.write(f"{met}")


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
                batch_size = st.selectbox("Page Size", options=[25, 50, 100],key="Volcano Plot")
            with bottom_menu[1]:
                total_pages = (
                    int(len(dataset) / batch_size) if int(len(dataset) / batch_size) > 0 else 1
                )
                current_page = st.number_input(
                    "Page", min_value=1, max_value=total_pages, step=1
                )
            with bottom_menu[0]:
                st.markdown(f"Page **{current_page}** of **{total_pages}** ")

            pages = split_frame(dataset, batch_size)
            pagination.dataframe(data=pages[current_page - 1], use_container_width=True)



        # Function to load and cache data
        #@st.cache(allow_output_mutation=True)
        @st.cache_resource
        def load_data(fasta_path, csv_path):
            record_dict = SeqIO.to_dict(SeqIO.parse(fasta_path, "fasta"))
            df = pd.read_csv(csv_path, sep="\t", header=0)
            return record_dict, df

        # Load data
        fasta_path = os.path.join("data", "protein_sequences.fasta")
        csv_path = os.path.join("data", "input_data.tsv")
        # Upload widgets
        record_dict, df = load_data(fasta_path, csv_path)


        # Filter out the 'fold_change' column from the options
        condition_columns = [col for col in df.columns[2:] if col != 'fold_change']
        # Sidebar Inputs
        #x_axis = st.sidebar.selectbox('X-Axis: condition A', df.columns[2:], index=0)
        #y_axis = st.sidebar.selectbox('Y-Axis: condition B', df.columns[2:], index=1)
        x_axis = st.sidebar.selectbox('Select condition for X-axis:', options=condition_columns, index=0)
        y_axis = st.sidebar.selectbox('Select condition for Y-axis:', options=condition_columns, index=1)
        cut_off = st.sidebar.slider('|fold change| > :', 0.0, 10.0, 2.0)
        gene_annotation = st.sidebar.text_input("Gene annotation contains:")

        






        # Process the data
        significant_genes = process_data(df, x_axis, y_axis, cut_off, record_dict)

        

        #plot = create_bokeh_plot(significant_genes, x_axis, y_axis, gene_annotation)

        # Display the plot
        #st.bokeh_chart(plot, use_container_width=True)


        # Create the plot
        #if not significant_genes.empty:
        #    plot = create_bokeh_plot(significant_genes, x_axis, y_axis, gene_annotation)
        #    st.bokeh_chart(plot, use_container_width=True)
        #else:
        #    st.warning("No data to display.")


        # Display the Bokeh plot
        if not significant_genes.empty:
            col1, col2 = st.columns([0.5,0.5])

            with col1.container(border=None):
                plot = create_bokeh_plot(significant_genes, x_axis, y_axis, gene_annotation)
                st.bokeh_chart(plot, use_container_width=True)

            with col2.container(border=None):
                # Filter data based on gene annotation input
                filtered_data = significant_genes[significant_genes['Annotation'].str.contains(gene_annotation, case=False, na=False)]

                # Display the filtered data using Ag-Grid
                # Enhanced display using HTML
                st.markdown("<h4 style='color: navy; text-align: center;'>Filtered Genes based on 'Annotation Contains':</h4>", unsafe_allow_html=True)
                paginate_filtered_data(filtered_data)
                #AgGrid(filtered_data)
        else:
            st.warning("No data to display.")

    with tab3:
        # Upload CSV/TSV file widget
        uploaded_file = st.file_uploader("Upload your CSV/TSV file", type=['csv', 'tsv'])

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file, delimiter='\t')

                col1, col2 = st.columns(2)
                with col1:
                    cont_columns = df.select_dtypes(include=[float, int]).columns
                    selected_columns = st.multiselect('Select columns for heatmap',
                                            cont_columns, 
                                            default=cont_columns.tolist())
                    colorscale = st.selectbox('Choose a colorscale', ['Viridis', 'Cividis', 'Plasma', 'Inferno', 'Magma', 'YlGnBu'])
                    
                    if selected_columns:
                        try:
                            corr_fig = create_corrmap(df, selected_columns, colorscale)
                            if corr_fig:
                                st.plotly_chart(corr_fig, use_container_width=True)
                        except ValueError as e:
                            st.error(e)
                with col2:
                    st.dataframe(df)

            except pd.errors.ParserError:
                st.error("Error parsing CSV file. Please check the file format.")
        else:
            st.write("Awaiting CSV/TSV file to be uploaded.")


def show_literature_search():
    st.subheader("Literature Search")

    tab1, tab2, tab3 = st.tabs(["Google Scholar", "Europe PMC", "Pubmed"])

    with tab1:
        st.subheader("Google Scholar")
        #show_googlescholar_results()
        # Check if there's a search term in the session state
        default_search_term = st.session_state.get('search_term', '')
        show_googlescholar_results(default_search_term)
        # Reset the session state after search
        st.session_state['search_term'] = ''

    with tab2:
        st.subheader("Europe PMC - Text Summarizer")
        show_europepmc_results()  # Wrapper function for show_europepmc_results()
    
    with tab3:
        st.subheader("Pubmed Summarizer")
        show_pubmed_results()


def show_googlescholar_results(default_search_term):
    # Code from the provided script
    # Include the functionality for literature search here
    # Modify the code as necessary to fit into the function

    #st.subheader("Literature Search")
    st.markdown("""
    Scraping relevant information of research papers from Google Scholar.
    """)

    # scraping function
    # creating final repository
    paper_repos_dict = {
                    'Paper Title' : [],
                    'Year' : [],
                    'Author' : [],
                    'Citation' : [],
                    'Publication site' : [],
                    'Url of paper' : [] }
    
    html_temp = """
                    <div style="background-color:{};padding:1px">
                    
                    </div>
                    """
    
    # adding information in repository
    def add_in_paper_repo(papername,year,author,cite,publi,link):
        paper_repos_dict['Paper Title'].extend(papername)
        paper_repos_dict['Year'].extend(year)
        paper_repos_dict['Author'].extend(author)
        paper_repos_dict['Citation'].extend(cite)
        paper_repos_dict['Publication site'].extend(publi)
        paper_repos_dict['Url of paper'].extend(link)
        #   for i in paper_repos_dict.keys():
        #     print(i,": ", len(paper_repos_dict[i]))
        #     print(paper_repos_dict[i])
        df = pd.DataFrame(paper_repos_dict)
        
        return df

    # headers
    headers = {'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'}
    url_begin = 'https://scholar.google.com/scholar?start={}&q='
    url_end = '&hl=en&as_sdt=0,5='
    # input
    col1, col2 = st.columns([3,1])
    with col1:
        #text_input = st.text_input("Search in Google Scholar", placeholder="What are you looking for?", disabled=False)
        text_input = st.text_input("Search in Google Scholar", value=default_search_term, placeholder="What are you looking for?", disabled=False) #<-- use Form instead of st.text_input

    with col2:
        total_to_scrap = st.slider("How many pages to scrap?", min_value=0, max_value=4, step=1, value=1)

    st.markdown(html_temp.format("rgba(55, 53, 47, 0.16)"),unsafe_allow_html=True)
    # create scholar url
    if text_input:
        text_formated = "+".join(text_input.split())
        input_url = url_begin+text_formated+url_end
        if input_url:
            response=requests.get(input_url,headers=headers)
            # st.info(input_url)
            total_papers = 10 * total_to_scrap
            for i in range (0,total_papers,10):
                # get url for the each page
                url = input_url.format(i)
                # function for the get content of each page
                doc = get_paperinfo(url, headers)

                # function for the collecting tags
                paper_tag,cite_tag,link_tag,author_tag = get_tags(doc)

                # paper title from each page
                papername = get_papertitle(paper_tag)

                # year , author , publication of the paper
                year , publication , author = get_author_year_publi_info(author_tag)

                # cite count of the paper 
                cite = get_citecount(cite_tag)

                # url of the paper
                link = get_link(link_tag)

                # add in paper repo dict
                final = add_in_paper_repo(papername,year,author,cite,publication,link)

                # use sleep to avoid status code 429
                sleep(20)
            
            final['Year'] = final['Year'].astype('int')
            final['Citation'] = final['Citation'].apply(cite_number).astype('int')

            with st.expander("Extracted papers"):
                st.dataframe(final)
                csv = convert_df(final)
                file_name_value = "_".join(text_input.split())+'.csv'
            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name=file_name_value,
                mime='text/csv',
            )

            # Plots
            col1, col2 = st.columns([2,1])

            with col1:
                with st.expander("Distribution of papers by year and citation", expanded=True):
                    size_button = st.checkbox('Set Citation as bubble size', value=True)
                    size_value = None
                    if size_button:
                        size_value = 'Citation'
                    final_sorted = final.sort_values(by='Year', ascending=True)
                    fig1 = px.scatter(
                        final_sorted, 
                        x="Year", 
                        color="Publication site",
                        size=size_value, 
                        log_x=True, 
                        size_max=60
                        )
                fig1.update_xaxes(type='category')
                st.plotly_chart(fig1, theme="streamlit", use_container_width=True)

            with col2:
                percentage_sites = {}
                sites = list(final_sorted['Publication site'])
                for i in sites:
                    percentage_sites[i] = sites.count(i)/len(sites)*100
                df_per = pd.DataFrame(list(zip(percentage_sites.keys(), percentage_sites.values())), columns=['sites', 'percentage'])
        
                fig2 = px.pie(
                        df_per, 
                        values="percentage", 
                        names="sites", 
                        )
                with st.expander("Percentage of publication sites", expanded=True):
                    st.plotly_chart(fig2, theme="streamlit", use_container_width=True)

    # Make sure to return any data or results that need to be displayed or used elsewhere


def show_europepmc_results():

    def display_results(literature):
        st.write("Search Results:")
        for title, summary in literature:
            st.write(f"Title: {title}")
            st.write(summary)
            st.write("---")
    
    #st.title("Gene Literature Full-Text Summarizer")

    # User input
    gene_name = st.text_input("Enter the name of the gene:")

    if st.button("Search") and gene_name:
        # Fetch literature from Europe PMC
        literature = fetch_literature(gene_name)
        # Display results
        display_results(literature)


def show_pubmed_results():
    # Streamlit interface
    #st.title('Literature Search App')

    query = st.text_input("Enter your search query", "")

    if st.button("Search", key="Pubmed") or 'pubmed_results' in st.session_state:
        if 'pubmed_results' not in st.session_state or st.session_state.query != query:
            # Fetch and store the results in session state
            st.session_state.pubmed_results = fetch_pubmed_abstracts(query)
            st.session_state.query = query
        
        # Plots
        tab1, tab2, tab3, tab4 = st.tabs(["Table","Pie chart", "Author Network", "TOP10 Bar plot"])
        with tab1:
            # Display the results using AgGrid
            display_results_in_aggrid(st.session_state.pubmed_results)

        with tab2:
            # Display Journal Distribution
            # Calculate journal distribution
            journal_distribution = calculate_journal_distribution(st.session_state.pubmed_results)

            # Create a DataFrame for the pie chart
            journal_distribution_df = pd.DataFrame(journal_distribution.items(), columns=["Journal", "Count"])

            # Create an interactive pie chart using Plotly Express
            fig = px.pie(
                journal_distribution_df,
                names="Journal",
                values="Count",
                title="Journal Distribution",
                labels={"Journal": "Journal", "Count": "Count"},
            )

            # Customize the layout of the chart
            fig.update_layout(
                legend_title_text="Journal",
                margin=dict(l=0, r=0, t=30, b=0),  # Adjust chart margin
            )

            # Render the Plotly figure
            st.plotly_chart(fig)

        with tab3:
            # Authors Network
            if st.session_state.pubmed_results:
                #st.subheader("Authors Network")
                authors_network = create_authors_network(st.session_state.pubmed_results)
                if authors_network:
                    pv_static(authors_network)

        with tab4:
            # Calculate author publication counts
            if st.session_state.pubmed_results:
                author_counts = calculate_author_publication_counts(st.session_state.pubmed_results)

                # Sort authors by publication counts in descending order and select the top 10
                top_authors = dict(sorted(author_counts.items(), key=lambda item: item[1], reverse=True)[:10])

                # Create a DataFrame for the barplot
                top_authors_df = pd.DataFrame(top_authors.items(), columns=["Author", "Publication Count"])

                # Create an interactive bar chart using Plotly Express
                fig = px.bar(
                    top_authors_df,
                    x="Author",
                    y="Publication Count",
                    color="Author",  # Assign different colors to authors
                    title="Top 10 Authors by Publication Count",
                    labels={"Author": "Authors", "Publication Count": "Publication Count"},
                )

                # Customize the layout of the chart
                fig.update_layout(
                    xaxis=dict(tickangle=45),  # Rotate labels for readability
                    xaxis_title="Authors",  # X-axis name
                    yaxis_title="Publication Count",  # Y-axis name
                    showlegend=False,  # Hide the legend
                )

                # Render the Plotly figure
                st.plotly_chart(fig)


def generate_report_ui(processed_data):
    """Displays the UI elements for report generation."""
    st.subheader("Report Generation")
    st.markdown("Generate a report of your analysis.")
    if st.button("Generate Report"):
        # Placeholder for report generation logic
        st.write("Report generation functionality to be implemented.")
        # Example: Display processed data or any other analysis result
        st.write(processed_data.head())


def feedback_ui():
    """Displays the UI elements for user feedback."""
    st.subheader("Feedback")
    feedback = st.text_area("Please provide your feedback about the app:")
    if st.button("Submit Feedback"):
        # Placeholder for feedback handling logic
        st.success("Thank you for your feedback!")


# main_app.py (Main function implementation)
def main():

    #set background
    #add_bg_from_local('image.png')

    # Sidebar navigation
    st.sidebar.title("Navigation")

    sections = ["Data Upload", "Data Preprocessing", "Build Plots", "Literature Search", "Report Generation", "Feedback"]
    
    selected_section = st.sidebar.radio("Go to", sections, key='nav')

   # Initialize variables to store uploaded data
    processed_data = None

    # Display the appropriate section
    if selected_section == "Data Upload":
        #Web App Title
        st.markdown('''
        # **RNA-Seq Analyzer**

        This is the **RNA-Seq EDA App** created using `Python` + `Streamlit`.

        ---
        ''')
    #**Credit:** App built by [Alexander Yemelin](https://www.linkedin.com/in/alexander-yemelin/)
        show_file_upload()
    elif selected_section == "Data Preprocessing":
        show_data_preprocessing()
    elif selected_section == "Build Plots":
        show_volcano_plot()
    elif selected_section == "Literature Search":
        #with st_lottie_spinner(lottie_json,height=400, width=400, key="literature"):
        #    time.sleep(1)
        show_literature_search()
    elif selected_section == "Report Generation":
        generate_report_ui(processed_data)
    elif selected_section == "Feedback":
        feedback_ui()
    

if __name__ == "__main__":
    #im = Image.open("favicon.ico")
    st.set_page_config(
        page_title="RNA-Seq EDA App", page_icon=":bar_chart:", 
        layout="wide",
        #im = Image.open("favicon.ico"), page_icon=":chart_with_upwards_trend:",
    )
    
    with st.spinner('Calculating...'):
    #with st_lottie_spinner(lottie_json,height=400, width=400):
        #time.sleep(1)
        main()
        #st.balloons()
    #from streamlit_lottie import st_lottie
    #st_lottie("https://assets5.lottiefiles.com/packages/lf20_V9t630.json", key="hello")



    with st.sidebar:
        #from streamlit_lottie import st_lottie
        #st_lottie("https://assets5.lottiefiles.com/packages/lf20_V9t630.json", key="hello")
        st.markdown("---")
        st.markdown(
            '<h6>Made in &nbsp<img src="https://streamlit.io/images/brand/streamlit-mark-color.png" alt="Streamlit logo" height="16">&nbsp by <a href="https://github.com/Alexyem1">@Alexyem1</a></h6>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div style="margin-top: 0.75em;"><a href="https://www.buymeacoffee.com/" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="41" width="174"></a></div>',
            unsafe_allow_html=True,
        )
        st.markdown("---")
        #st.info('Credit: Created by [Alexander Yemelin](https://www.linkedin.com/in/alexander-yemelin/)')
        st.caption('''Every exciting data science journey starts with a dataset. Please upload a CSV file. Once we have the data in hand, we'll dive into understanding it and have some fun exploring it.''')




