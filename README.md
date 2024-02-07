# Streamlit RNA-Seq Analyzer App
(Streamlit web app for RNA-Seq data analysis)

## Description
"Introducing the RNA-Seq Analyzer: A Streamlit-Enhanced Tool for RNA-Seq Data Analysis"
The RNA-Seq Analyzer revolutionizes the exploration of RNA-Seq data through a user-friendly, web-based platform. Developed with Streamlit, it simplifies the exploratory data analysis process, enabling users to upload, filter, visualize, and analyze their datasets with ease. Compatible with both CSV and Excel formats, this tool is tailored to meet the needs of a diverse user base, from data analysts to academic researchers.

Access the tool directly via https://github.com/Alexyem1/RNASeq-Analyzer/archive/refs/heads/main.zip and embark on a seamless analytical journey, leveraging the RNA-Seq Analyzer's capabilities to unlock valuable insights from your data. Designed to enhance accessibility and efficiency in RNA-Seq data analysis, this tool is a valuable asset for the scientific and research community.

<div style="text-align:center"><img src ="https://github.com/Alexyem1/RNASeq-Analyzer/blob/main/workflow.png" /></div>

## Features

- **Data Upload**: Upload CSV or TSV files for analysis, containing FPKM values or fold change and p-values.
- **Data Preprocessing**: Filter and examine specific columns in your dataset, remove missing values and duplicates, and perform descriptive statistics analysis
- **Data Analysis**: Create interactive volcano plot, scatter plot, clustergram and perform scientific literature search through data fetching from publicly accessible databases, including PubMed, Europe PMC, and Google Scholar.
- **Report Generation**: Automatic report generating feature summarizes created plots in their current states in a separated report that can be exported and downloaded as .pdf file.

## How to Use

1. **Upload Data**: Upload your dataset in CSV or TSV format.
2. **Data Preprocessing**: View and select specific columns and rows from your dataset.
3. **Data Visualization**: Select from various plot types for data visualization.
4. **Report Generation**: After analysis, users can download the output plots in a collective manner. The result is a customized data report showcasing the outcomes of the data analysis based on selected plots. This report is automatically generated once the user start data analysis by analyzing different plots and automatically updated with every change of the corresponding plots.

# Local Installation - Reproducing this web app
To recreate this web app on your own computer, do the following.

### Create conda environment
Firstly, we will create a conda environment called *rnaseq_analyzer*
```
conda create -n rnaseq_analyzer python=3.8
```
Secondly, we will login to the *rnaseq_analyzer* environment
```
conda activate rnaseq_analyzer
```
### Install prerequisite libraries

Download requirements.txt file

```
wget https://github.com/Alexyem1/RNASeq-Analyzer/blob/main/requirements.txt

```

Pip install libraries
```
pip install -r requirements.txt
```

###  Download and unzip contents from GitHub repo

Download and unzip contents from https://github.com/Alexyem1/RNASeq-Analyzer/archive/refs/heads/main.zip

###  Launch the app

```
streamlit run main_app.py
```


---

Contributions to improve the app are welcome! Feel free to fork the repository and submit a pull request.
