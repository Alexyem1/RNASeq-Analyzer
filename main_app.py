# main_app.py

#UI and Interaction Functions:
    # show_file_upload: Manages the UI for file upload.
    # show_missing_data: Displays missing data information in the UI.
    # show_data_preprocessing: Handles the UI for data preprocessing.
    # show_data_analysis: Specifically for displaying plots and literature search results in the UI.
    # show_literature_search, show_googlescholar_results, show_europepmc_results, show_pubmed_results: Manage the UI for literature search and results display.
    # generate_report_ui, feedback_ui: Handle UI elements for report generation and feedback.
# Core Functionalities:
    # main: The main function to run the app, which orchestrates the flow and interactions.

# Import necessary modules
import streamlit as st
import streamlit.components.v1 as components
from streamlit_extras.stateful_button import button
from streamlit_lottie import st_lottie
from streamlit_plotly_events import plotly_events

from utils import add_spacer, fetch_pubmed_abstracts, display_results_in_aggrid, calculate_journal_distribution, calculate_author_publication_counts,create_authors_network
from utils import read_file, filter_dataframe, paginate_df, process_data, create_bokeh_plot, create_corrmap, fetch_literature, plot_to_bytes
from utils import get_paperinfo, get_tags, get_papertitle, get_citecount, get_link, get_author_year_publi_info, cite_number, convert_df, create_plotly_volcano_plot

import pandas as pd
import numpy as np

from stvis import pv_static
import requests
from time import sleep
import missingno as msno
import matplotlib.pyplot as plt
import plotly.express as px
from Bio import SeqIO
import os
import jinja2
import pdfkit
#from PIL import Image
from datetime import datetime
import pytz
import json
import warnings
warnings.filterwarnings("ignore")

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
        msno.matrix(df, ax=ax, sparkline=False)

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
                def generate_table_data():
                    if 'uploaded_data' in st.session_state and not st.session_state['uploaded_data'].empty:
                        df = st.session_state['uploaded_data']
                        return pd.DataFrame({
                            "Description": ["Number of Rows", "Number of Columns", "Number of Duplicated Rows", "Number of Rows with Missing Values"],
                            "Value": [
                                df.shape[0],
                                df.shape[1],
                                sum(df.duplicated()),
                                df.isnull().sum().sum()
                            ]
                        })
                    else:
                        return pd.DataFrame()
                    
                
                def convert_df_to_tsv(df):
                    """Convert DataFrame to TSV string."""
                    return df.to_csv(sep='\t', index=False)


                # Initialize button state in session state
                if 'remove_duplicates_clicked' not in st.session_state:
                    st.session_state['remove_duplicates_clicked'] = False

                if 'remove_missing_clicked' not in st.session_state:
                    st.session_state['remove_missing_clicked'] = False

                if "duplicate_row_count" not in st.session_state:
                    st.session_state["duplicate_row_count"] = None
                # Use the duplicated() function to identify duplicate rows
                duplicates = df[df.duplicated()]
                duplicate_row_count =  duplicates.shape[0]
                st.session_state["duplicate_row_count"] = duplicate_row_count

                if "missing_value_row_count" not in st.session_state:
                    st.session_state["missing_value_row_count"] = None
                missing_value_row_count = df[df.isna().any(axis=1)].shape[0]
                st.session_state["missing_value_row_count"] = missing_value_row_count

                # Remove duplicates button
                #if button("Remove duplicates", key="button1"):
                if duplicate_row_count != 0 and not st.session_state['remove_duplicates_clicked']:
                    if st.button("Remove duplicates"):
                        st.session_state['remove_duplicates_clicked'] = True
                        df.drop_duplicates(keep='first', inplace=True)
                        st.session_state['uploaded_data'] = df
                        st.session_state["duplicate_row_count"] = df[df.duplicated()].shape[0]
                        st.success("Duplicates removed.")

                # Remove rows with missing values button
                #if button("Remove rows with missing values", key="button2"):
                if missing_value_row_count != 0 and not st.session_state['remove_missing_clicked']:
                    if st.button("Remove rows with missing values"):
                        st.session_state['remove_missing_clicked'] = True
                        df.dropna(inplace=True)
                        st.session_state['uploaded_data'] = df
                        st.session_state["missing_value_row_count"] = df.isnull().sum().sum()
                        st.success("Rows with missing values removed.")
                



                # Update and display the summary table
                table_df = generate_table_data()
                table_df.set_index('Description', inplace=True)
                st.dataframe(table_df)


                if st.session_state['remove_duplicates_clicked'] or st.session_state['remove_missing_clicked']:
                    st.success("Data is ready for plotting!")
                    # Button for downloading the processed data as TSV
                    df = st.session_state['uploaded_data']
                    tsv_string = convert_df_to_tsv(df)
                    st.download_button(
                        label="📥 Download data as TSV",
                        data=tsv_string,
                        file_name='processed_data.tsv',
                        mime='text/tsv')





            with col2:
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


def show_data_analysis():
    #st.title('Enhanced Volcano Plot Visualization with Streamlit and Dash')

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Volcano Plot", "Scatter Plot", "Clustergram", "Literature", "Report generation"])

    with tab1:
        st.subheader("Volcano Plot")

        #st.write("Upload your data (CSV or TSV format) to visualize the volcano plot.")

        # File uploader that accepts both CSV and TSV files
        uploaded_file = st.file_uploader("Upload your CSV/TSV file", type=['csv', 'tsv'], key="Volcano_Plot_Uploader")

        if uploaded_file is not None:
            file_type = uploaded_file.name.split('.')[-1]
            if file_type == 'csv':
                df = pd.read_csv(uploaded_file)
            elif file_type == 'tsv':
                df = pd.read_csv(uploaded_file, sep='\t')

            sel_columns = df.select_dtypes(include=[np.number, object]).columns

            col1, col2, col3, col4 = st.columns(4)
            with col1.container(border=True):
                sel_col_id = st.multiselect('Select GeneID column:', sel_columns, key="sel_col_id")
            with col2.container(border=True):
                sel_col_P = st.multiselect('Select P-values column:', sel_columns, key="sel_col_P")
            with col3.container(border=True):
                sel_col_FC = st.multiselect('Select FC values column:', sel_columns, key="sel_col_FC")
            with col4.container(border=True):
                sel_col_ann = st.multiselect('Select Annotation column:', sel_columns, key="sel_col_ann")
            
            col5, col6, col7 = st.columns(3)
            with col5.container(border=None):
                p_value_filter = st.slider('P-value Filter', 0.0, 1.0, 0.05, 0.01)
            with col6.container(border=None):
                fold_change_filter = st.slider('Fold Change Filter', 0.0, 10.0, 1.0, 0.1)
            with col7.container(border=None):
                gene_annotation = st.text_input("Gene annotation contains:", key="gene_annotation")



            if sel_col_P and sel_col_FC and sel_col_ann and sel_col_id:
                plot = create_plotly_volcano_plot(df, sel_col_P[0], sel_col_FC[0], sel_col_ann[0],sel_col_id[0], gene_annotation, p_value_filter, fold_change_filter)
                st.session_state["volcano_plot"] = plot
                selected_data = plotly_events(plot, click_event=True, select_event=True)
                #st.plotly_chart(plot, use_container_width=True)

                if selected_data:
                    selected_indices = [point["pointIndex"] for point in selected_data]
                    #st.write("Selected Indices:", selected_indices)
                    selected_data = df.iloc[selected_indices]
                    st.dataframe(selected_data)
                else:
                    st.write("No genes selected")
        else:
            st.write("Awaiting CSV/TSV file to be uploaded.")


        
    with tab2:
        st.subheader("Scatter Plot")
        #st.write("Upload your data (CSV or TSV format) to visualize the scatter plot.")
        #Pagination module
        #@st.cache_data(show_spinner=False) <-- not sure wether I should use it here
        def split_frame(input_df, rows):
            df = [input_df.loc[i : i + rows - 1, :] for i in range(0, len(input_df), rows)]
            return df

        def paginate_filtered_data(dataset):
            top_menu = st.columns(3)
            with top_menu[0]:
                st.metric(label="Gene count", value=len(filtered_data))


            with top_menu[1]:
                sort_field = st.selectbox("Sort By", options=dataset.columns)
            with top_menu[2]:
                sort_direction = st.radio(
                        "Direction", options=["⬆️", "⬇️"], horizontal=True, key="sort_direction_key"
                    )
            dataset = dataset.sort_values(
                    by=sort_field, ascending=sort_direction == "⬆️", ignore_index=True
                )
            pagination = st.container()

            bottom_menu = st.columns((4, 1, 1))
            with bottom_menu[2]:
                batch_size = st.selectbox("Page Size", options=[25, 50, 100],key="Volcano_Plot")
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

            try:
                # Attempt to display the page. current_page is assumed to be defined.
                pagination.dataframe(data=pages[current_page - 1], use_container_width=True)
            except IndexError:
                # This block is executed if an IndexError occurs in the try block.
                st.error("No data available to display.")


        # Function to load and cache data
        @st.cache_resource
        def load_data(fasta_path):
            record_dict = SeqIO.to_dict(SeqIO.parse(fasta_path, "fasta"))
            return record_dict

        # Load data
        fasta_path = os.path.join("data", "protein_sequences.fasta")
        # Upload widgets
        record_dict = load_data(fasta_path)


        # File uploader that accepts both CSV and TSV files
        scatterplot_df = st.file_uploader("Upload your CSV/TSV file", type=['csv', 'tsv'], key = "Scatter_Plot_Uploader")

        if scatterplot_df is not None:
            file_type = scatterplot_df.name.split('.')[-1]
            if file_type == 'csv':
                sp_df = pd.read_csv(scatterplot_df)
            elif file_type == 'tsv':
                sp_df = pd.read_csv(scatterplot_df, sep='\t')
            
            if "sp_df" not in st.session_state:
                st.session_state["sp_df"] = None
            st.session_state["sp_df"] = sp_df

            sel_columns = sp_df.select_dtypes(include=[np.number]).columns
            if "sp_sel_columns" not in st.session_state:
                st.session_state["sp_sel_columns"] = None

            st.session_state["sp_sel_columns"] = sel_columns


            # Filter out the 'fold_change' column from the options
            #condition_columns = [col for col in df.columns[2:] if col != 'fold_change']
            
            # Sidebar Inputs
            #x_axis = st.sidebar.selectbox('X-Axis: condition A', df.columns[2:], index=0)
            #y_axis = st.sidebar.selectbox('Y-Axis: condition B', df.columns[2:], index=1)
            col1, col2, col3, col4 = st.columns([0.25,0.25,0.25,0.25])
            with col1:
                x_axis = st.selectbox('Select condition for X-axis:', options=st.session_state.sp_sel_columns,  key=("selx"))
            with col2:
                y_axis = st.selectbox('Select condition for Y-axis:', options=st.session_state.sp_sel_columns,  key=("sely"))
            with col3:
                cut_off = st.slider('|fold change| > :', 0.0, 10.0, 2.0)
            with col4:
                gene_annotation = st.text_input("Gene annotation contains:")

            # Process the data
            significant_genes, df_processed = process_data(st.session_state.sp_df, x_axis, y_axis, cut_off, record_dict)

            # Display the Bokeh plot
            if not significant_genes.empty:
                col1, col2 = st.columns([0.5,0.5])

                with col1.container(border=None):
                    plot = create_bokeh_plot(df_processed,significant_genes, x_axis, y_axis, gene_annotation)
                    st.session_state["scatter_plot"] = plot
                    
                    st.bokeh_chart(plot, use_container_width=True)

                with col2.container(border=None):
                    # Filter data based on gene annotation input
                    filtered_data = significant_genes[significant_genes['Annotation'].str.contains(gene_annotation, case=False, na=False)]

                    # Display the filtered data using Ag-Grid
                    # Enhanced display using HTML
                    st.markdown("<h4 style='color: navy; text-align: center;'>List of filtered genes:</h4>", unsafe_allow_html=True)
                    paginate_filtered_data(filtered_data)
                    #AgGrid(filtered_data)
            else:
                st.warning("No data to display.")
        else:
            st.write("Awaiting CSV/TSV file to be uploaded.")

    with tab3:
        st.subheader("Clustergram")
        # Upload CSV/TSV file widget
        uploaded_file = st.file_uploader("Upload your CSV/TSV file", type=['csv', 'tsv'], key="Clustergram_Uploader")

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file, delimiter='\t')

                col1, col2 = st.columns(2)
                with col1:
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        cont_columns = df.select_dtypes(include=[float, int]).columns

                        selected_columns = st.multiselect('Select columns for heatmap',
                                            cont_columns, 
                                            default=cont_columns.tolist())
                    with col_b:
                        colorscale = st.selectbox('Choose a colorscale', ['Viridis', 'Cividis', 'Plasma', 'Inferno', 'Magma', 'YlGnBu'])

                    with col_c:
                        # Selector for correlation method
                        correlation_method = st.selectbox(
                            "Select Correlation Method",
                            ["pearson", "kendall", "spearman"],
                            index=0
                        )

                
                    
                    if selected_columns:
                        try:
                            corr_fig = create_corrmap(df, selected_columns, colorscale, correlation_method)
                            st.session_state["clustergram_plot"] = corr_fig
                            if corr_fig:
                                st.plotly_chart(corr_fig, use_container_width=True)
                        except ValueError as e:
                            st.error(e)
                with col2:
                    df.set_index(str(df.columns[0]), inplace=True)
                    st.dataframe(df)

            except pd.errors.ParserError:
                st.error("Error parsing CSV file. Please check the file format.")
        else:
            st.write("Awaiting CSV/TSV file to be uploaded.")
    with tab4:
        show_literature_search()
    
    with tab5:
        generate_report_ui()


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
            # Add a checkbox to enable/disable network calculation
            is_active = st.checkbox('Enable Authors Network Calculation', value=False)

            if is_active and st.session_state.pubmed_results:
                authors_network = create_authors_network(st.session_state.pubmed_results)
                if authors_network:
                    pv_static(authors_network)
            else:
                st.write("Authors Network Calculation is disabled.")

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


def generate_report_ui():

    st.header("Report Generation")
    # Get time
    today = datetime.now(tz=pytz.timezone("Europe/Berlin"))
    dt_string = today.strftime("%d %B %Y %I:%M:%S %p (CET%z)")

    st.info("Omitting any steps in the process will result in a report that displays the header, but lacks the plots, as these were not executed.")

    # Converting all the plots to output to buffer with BytesIO

    all_plots_bytes = {}

    for key in ['volcano_plot', 'clustergram_plot']:
        if key in st.session_state:
            if st.session_state[key] is not None:
                to_bytes = plot_to_bytes(st.session_state[key], graph_module='plotly', format='png')
                all_plots_bytes[key] = to_bytes
            else:
                all_plots_bytes[key] = None
        else:
            all_plots_bytes[key] = None
        

    for key in ['scatter_plot']:
        if key in st.session_state:
            if st.session_state[key] is not None:
                to_bytes = plot_to_bytes(st.session_state[key], graph_module='bokeh', format='png')
                all_plots_bytes[key] = to_bytes
            else:
                all_plots_bytes[key] = None
        else:
            all_plots_bytes[key] = None



    templateLoader = jinja2.FileSystemLoader(searchpath="accessory_files/")
    templateEnv = jinja2.Environment(loader=templateLoader)
    TEMPLATE_FILE = "output_report_template.html"
    template = templateEnv.get_template(TEMPLATE_FILE)


    outputText = template.render(date = dt_string,
                                volplot = all_plots_bytes['volcano_plot'],
                                clustergram=all_plots_bytes['clustergram_plot'],
                                scatterplot=all_plots_bytes['scatter_plot']
                                )
    html_file = open("RNASeq_Analyzer_report.html", 'w')
    html_file.write(outputText)
    html_file.close()


    HTML_inapp = open("RNASeq_Analyzer_report.html", 'r', encoding='utf-8')
    source_code = HTML_inapp.read()
    components.html(source_code, height = 900, scrolling=True)

    #pdf_out = pdfkit.from_string(outputText, False)
    # Set the path to the wkhtmltopdf executable
    path_wkhtmltopdf = r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'  # Adjust the path if different

    # Create a configuration object with the specified path
    config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)

    # Use the configuration in your pdfkit call
    pdf_out = pdfkit.from_string(outputText, False, configuration=config)
    st.download_button("Download RNASeq Analyzer report as PDF here", data=pdf_out, file_name="RNASEQ_Analyzer_report.pdf", mime='application/octet-stream')

    
def feedback_ui():
    a, b = st.columns([0.9, 1.4], gap='large')
    a.image('IGI_Berkeley.jpg')
    a.markdown("""<span style="font-size:20px;">Innovative Genomics Institute (IGI) | Arkin Lab </span>""", unsafe_allow_html=True)

    a.markdown('##### If you have any questions or feedback, please feel free to contact me:')
    a.markdown("""<span style="font-size:18px;">**PhD. Alexander Yemelin**<br>Bioengineering and Computational Biology<br>✉️ yemelin.alexander@gmail.com<br>🖥️  https://github.com/Alexyem1</span>""", unsafe_allow_html=True)
    b.markdown("""<div style="width: 100%"><iframe src="https://www.google.com/maps/embed?pb=!1m14!1m8!1m3!1d12597.68161349268!2d-122.266984!3d37.87385!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x80857e9fb44b54b1%3A0xf41e987fe6d532ef!2sInnovative%20Genomics%20Institute!5e0!3m2!1sen!2sus!4v1705688695710!5m2!1sen!2sus" width="600" height="450" style="border:0;" allowfullscreen="" loading="lazy" referrerpolicy="no-referrer-when-downgrade"></iframe></div>""", unsafe_allow_html=True)


# main_app.py (Main function implementation)
def main():

    page = st.sidebar.radio("Go to: ", tuple(PAGES.keys()), format_func=str.capitalize)

    PAGES[page]()

PAGES = {
    "Data Upload": show_file_upload,
    "Data Preprocessing": show_data_preprocessing,
    "Data analysis": show_data_analysis,
    "Feedback": feedback_ui,
}

if __name__ == "__main__":
    #im = Image.open("favicon.ico")
    st.set_page_config(
        page_title="RNA-Seq EDA App", page_icon=":bar_chart:", 
        layout="wide",
        #im = Image.open("favicon.ico"), page_icon=":chart_with_upwards_trend:",
    )


    
    with st.spinner('Calculating...'):
        st.title("Streamlit RNA-Seq Analyzer")
        main()



    with st.sidebar:
        with open("./accessory_files/animation.json", "r",errors='ignore') as f:
            animation = json.load(f)
        st_lottie(animation)
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
        st.caption('''Every exciting data science journey starts with a dataset. Please upload a CSV/TSV file. Once we have the data in hand, we'll dive into understanding it and have some fun exploring it.''')




