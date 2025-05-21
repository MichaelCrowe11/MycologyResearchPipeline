
import streamlit as st
import pandas as pd

@st.cache_data
def load_data():
    df_species = pd.read_csv("CMID_v2_Expanded_Species.csv")
    df_citations = pd.read_csv("CMID_v2_PubMed_Citations.csv")
    df_compounds = pd.read_csv("CMID_v2_Compound_Structure_Matrix.csv")
    return df_species, df_citations, df_compounds

df_species, df_citations, df_compounds = load_data()

st.title("CMID v2.0 | Comprehensive Mushroom Intelligence Dashboard")

species_list = df_species["Common Name"].unique()
selected_species = st.sidebar.selectbox("Select a Mushroom Species", species_list)

st.header(f"Species Overview: {selected_species}")
species_info = df_species[df_species["Common Name"] == selected_species].T
st.dataframe(species_info)

st.subheader("Relevant PubMed Citations")
selected_sci_name = df_species[df_species["Common Name"] == selected_species]["Scientific Name"].values[0]
filtered_citations = df_citations[df_citations["Species"] == selected_sci_name]
st.dataframe(filtered_citations[["Title", "Authors", "Year", "Journal", "DOI"]])

st.subheader("Associated Compounds & Structures")
filtered_compounds = df_compounds[df_compounds["Associated Species"] == selected_sci_name]
st.dataframe(filtered_compounds)
