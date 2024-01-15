# main_app.py

# Import necessary modules
import streamlit as st
from utils import add_spacer, fetch_pubmed_abstracts, display_results_in_aggrid, calculate_journal_distribution, calculate_author_publication_counts,create_authors_network
import tempfile
from st_aggrid import AgGrid
import pandas as pd
from faker import Faker
import random
import io
from stvis import pv_static
from bokeh.models import WheelZoomTool
import plotly.graph_objects as go
import numpy as np
import dash_bio as dashbio
from scipy.cluster import hierarchy as sch
from scipy.spatial.distance import pdist
# Define UI functions for each section of the app

#background image for the app
import base64
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

    
def show_file_upload():
    uploaded_file = st.file_uploader("Upload your CSV/TSV file", type=['csv', 'tsv'], key="Data Upload")

    if uploaded_file is not None:
        df = read_file(uploaded_file)

        # Store the DataFrame in session state
        st.session_state['uploaded_data'] = df
        # Display the DataFrame using AgGrid
        #AgGrid(df)
        #st.dataframe(df)
        st.write("Data overview:")
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



# Add similar functions for other sections like show_data_preprocessing() and show_go_terms_visualization()
# ... [Continue with other UI functions] ...
######################################################################################################################
import pandas as pd
import streamlit as st
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

# ... [Your existing imports and code] ...

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

    return df

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
            "Page", min_value=1, max_value=total_pages, step=1
        )
    with bottom_menu[0]:
        st.markdown(f"Page **{current_page}** of **{total_pages}** ")

    pages = split_frame(dataset, batch_size)
    pagination.dataframe(data=pages[current_page - 1], use_container_width=True)


import missingno as msno
import matplotlib.pyplot as plt

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


import plotly.express as px
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



import dash_bio as dashbio
from streamlit_plotly_events import plotly_events
import streamlit as st
from Bio import SeqIO
import pandas as pd
from bokeh.plotting import figure
#from bokeh.models import ColumnDataSource, HoverTool, CrosshairTool
from bokeh.models import ColumnDataSource, OpenURL, TapTool, HoverTool, CrosshairTool
import os
import numpy as np
from st_aggrid import AgGrid
import plotly.graph_objects as go


# Cached function for loading data
@st.cache_data
def load_volcano_data(url):
    return pd.read_csv(url)

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

        # Function to process data
        def process_data(df, x_axis, y_axis, cutoff):
            # Calculate fold change
            #df['fold_change'] = df.apply(lambda row: (row[x_axis] / row[y_axis]) if row[y_axis] != 0 else np.inf, axis=1)

            # Filter for significant genes based on cutoff
            #significant_genes = df[np.abs(df['fold_change']) > cutoff]

            #return significant_genes

            df['fold_change'] = df.apply(
                lambda row: (row[x_axis] / row[y_axis]) if row[y_axis] != 0 else float('inf') if row[x_axis] > 0 else float('-inf'),
                axis=1
            )
            significant_genes = df[(abs(df['fold_change']) > cutoff) & (df[x_axis] + df[y_axis] > 10)]

            if not significant_genes.empty:
                significant_genes.loc[:, 'x_values'] = significant_genes[x_axis]
                significant_genes.loc[:, 'y_values'] = significant_genes[y_axis]
                significant_genes.loc[:, 'seq'] = significant_genes['GeneID'].apply(
                    lambda id: str(record_dict[id].seq) if id in record_dict else 'N/A'
                )
            return significant_genes






        # Process the data
        significant_genes = process_data(df, x_axis, y_axis, cut_off)

        # Create Bokeh plot
        def create_bokeh_plot(significant_genes, x_axis, y_axis, gene_annotation):
            significant_genes['x_values'] = significant_genes[x_axis]
            significant_genes['y_values'] = significant_genes[y_axis]
            significant_genes['color'] = 'blue'
            
            if gene_annotation:
                significant_genes.loc[significant_genes['Annotation'].str.contains(gene_annotation, case=False, na=False), 'color'] = 'red'

            source = ColumnDataSource(significant_genes)
            
            p = figure(title=f"{x_axis} vs {y_axis}", tools="pan,box_zoom,wheel_zoom,reset,save,tap")
            p.toolbar.active_scroll = p.select_one(WheelZoomTool)
            p.xaxis.axis_label = x_axis
            p.yaxis.axis_label = y_axis
            p.yaxis.axis_label_text_font_size = '14pt'
            p.xaxis.axis_label_text_font_size = '14pt'
            p.yaxis.major_label_text_font_size = '12pt'
            p.xaxis.major_label_text_font_size = '12pt'
            p.title.text_font_size = '16pt'
            p.circle('x_values', 'y_values', source=source, size=7, color='color', line_color=None)

            # Add tools to the plot
            url = "http://papers.genomics.lbl.gov/cgi-bin/litSearch.cgi?query=@seq&Search=Search"
            taptool = p.select(type=TapTool)
            taptool.callback = OpenURL(url=url)
            hover = HoverTool(tooltips=[("GeneID", "@GeneID"), ("Fold Change", "@fold_change"), ("Annotation", "@Annotation")])
            p.add_tools(hover)
            p.add_tools(CrosshairTool())

            return p

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
        # Function to create the correlation Clustergram
        def create_corrmap(df, cols, colorscale):
            # Ensure DataFrame is numeric and compute correlation matrix
            df_numeric = df[cols].select_dtypes(include=[np.number]) 
            df_corr = df_numeric.corr() # Compute correlation matrix

            # Simplified Clustergram call
            try:
                fig = dashbio.Clustergram(
                    data=df_corr.values,
                    column_labels=df_corr.columns.tolist(),
                    row_labels=df_corr.index.tolist(),
                    color_map=colorscale
                
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



# Note: The actual implementation of the function will require the existing Streamlit (st) context and the helper functions from the utils.py file.
# The provided code is a conceptual representation of the required modifications.




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




#Literature search module, modified version of the code from https://github.com/nainiayoub/scholar-scrap:
import streamlit as st
import requests
import urllib
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
from time import sleep
from utils import get_paperinfo, get_tags, get_papertitle, get_citecount, get_link, get_author_year_publi_info, cite_number, convert_df
# ... other imports ...

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


import streamlit as st
import requests
from transformers import T5Tokenizer, T5ForConditionalGeneration
import requests
from xml.etree import ElementTree as ET

# Streamlit app
def show_europepmc_results():

    def fetch_literature(drug_name):
        url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query={drug_name}&format=json&resultType=core"

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

            for i, article in enumerate(articles[:10]):  # Limit to first 10 articles
                print(article.get('id'))
                progress_bar.progress((i+1)/10)
                status_text.text(f"Processing article {i+1}/10")

                try:
                    full_text = get_full_text(article.get('id'))
                    print(full_text)
                    if full_text:
                        # Ensure the drug name is included in the summary
                        relevant_sentences = filter_sentences(full_text, drug_name)
                        relevant_text = " ".join(relevant_sentences)
                        # Summarize the filtered text
                        inputs = tokenizer.encode("summarize: " + relevant_sentences, return_tensors="pt", max_length=512, truncation=True)
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

    def filter_sentences(text, drug_name):
        sentences = text.split('.')
        return [sentence.strip() + '.' for sentence in sentences if drug_name.lower() in sentence.lower()]

    def get_full_text_old(article_id):
        # Function to retrieve the full text of an article
        full_text_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{article_id}/fullTextXML"
        response = requests.get(full_text_url)
        if response.status_code == 200:
            return response.text  # Or parse XML to extract relevant sections
        return None
    


    def get_full_text(article_id):
        full_text_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{article_id}/fullTextXML"

        try:
            response = requests.get(full_text_url)
            if response.status_code == 200:
                # Parse the XML response
                tree = ET.fromstring(response.content)
                # Extract the full text from the XML
                # Note: This depends on the structure of the XML response
                # You may need to adjust the following line to match the actual XML structure
                full_text = tree.find('.//fullText').text
                return full_text
            else:
                print(f"Failed to fetch full text for article ID {article_id}: HTTP {response.status_code}")
                return None
        except Exception as e:
            print(f"Error occurred while fetching full text for article ID {article_id}: {e}")
            return None


    def display_results(literature):
        st.write("Search Results:")
        for title, summary in literature:
            st.write(f"Title: {title}")
            st.write(summary)
            st.write("---")
    
    #st.title("Drug Literature Full-Text Summarizer")

    # User input
    drug_name = st.text_input("Enter the name of the drug:")

    if st.button("Search") and drug_name:
        # Fetch literature from Europe PMC
        literature = fetch_literature(drug_name)
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


# ... [Add other functions and the main function as previously described] ...

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

# ... [Continue with the main function implementation] ...

#import time
#import requests

#import streamlit as st
#from streamlit_lottie import st_lottie_spinner
#
#def load_lottieurl(url: str):
#    r = requests.get(url)
#    if r.status_code != 200:
#        return None
#    return r.json()

#lottie_url_old = "https://assets5.lottiefiles.com/packages/lf20_V9t630.json"
#lottie_url ="https://assets1.lottiefiles.com/packages/lf20_vykpwt8b.json"
#lottie_json = load_lottieurl(lottie_url)



# main_app.py (Main function implementation)

def main():
    #set background
    #add_bg_from_local('image.png')
    # Sidebar navigation
    st.sidebar.title("Navigation")

    sections = ["Data Upload", "Data Preprocessing", "Build Plots", "Literature Search", "Report Generation", "Feedback"]
    
    selected_section = st.sidebar.radio("Go to", sections, key='nav')

   # Initialize variables to store uploaded data
    raw_counts_data = metadata_data = processed_data = None

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

#prevent leaving the website by asking, later to implement
#    st.markdown("""
#    <script>
#        window.addEventListener('beforeunload', function (e) {
#            e.preventDefault();
#            e.returnValue = '';
#        });
#    </script>
#    """, unsafe_allow_html=True)
    
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




