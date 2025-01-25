import os
from langchain_community.vectorstores.chroma import Chroma
from connectdb import connect_to_postgresql, load_chroma
from brain import get_index_for_files, get_embedding
import streamlit as st
import openai
import chromadb
from sentence_transformers import SentenceTransformer
import pandas as pd
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

@st.cache_data
def create_vectordb(files, filenames):
    if not files or not filenames:
        st.error("No files provided.")
        return None
    if len(files) != len(filenames):
        st.error("Mismatch in the number of files and filenames.")
        return None
    try:
        with st.spinner("Creating vector database..."):
            vector_data = [file.getvalue() for file in files]
            if not vector_data:
                st.error("No data found in the uploaded files.")
                return None
            flag = get_index_for_files(vector_data, filenames)
            return flag
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

connection = connect_to_postgresql()

if connection:
    df = query_qas()
    # Close the database connection
    connection.close()
else:
    st.warning("Failed to connect to PostgreSQL.")

# File uploader for PDF and text files
file_types = ["pdf", "txt"]
uploaded_files = st.file_uploader("Upload files", type=file_types, accept_multiple_files=True)

# Handling uploaded files
if uploaded_files:
    file_names = [file.name for file in uploaded_files]

    # Store uploaded files in session_state
    st.session_state.uploaded_files = uploaded_files

    # Create and store flag in session_state
    st.session_state.flag = create_vectordb(uploaded_files, file_names)

collection = load_chroma()

# Get the data from the collection
data = collection.get()

# Convert the data to a DataFrame
df = pd.DataFrame(data)

# Sort DataFrame by index in descending order (reverse order)
df = df.sort_index(ascending=False)

# Add a checkbox column to select rows
select_all_checked  = st.checkbox('Select All', key='select_all')

# Display 'ids' in the multiselect widget
selected_indices = st.multiselect('Select items to delete:', df['ids'].tolist())

# Call collection.delete() with selected IDs
if st.button("Delete Selected Rows"):
    if select_all_checked == True:
        collection.delete(ids=[])
        st.success("Deleted all rows successfully!")
    else:
        collection.delete(ids=selected_indices)
        st.success("Selected rows deleted successfully!")
    selected_indices = []
        
# Add a search input field
search_query = st.text_input('Search:', '')

# Filter data based on search query across all columns
filtered_data = df[df.apply(lambda row: any(str(val).lower().find(search_query.lower()) != -1 for val in row), axis=1)]
# Display paginated and filtered data
page_num = st.number_input('Enter page number:', min_value=1, max_value=len(filtered_data)//10 + 1, value=1)
start_idx = (page_num - 1) * 10
end_idx = min(start_idx + 50, len(filtered_data))

# Display paginated data from end to start
paginated_data = filtered_data.iloc[start_idx:end_idx]
if len(paginated_data) > 0:
    # Display paginated data in a Streamlit DataFrame
    st.dataframe(paginated_data, height=500)
else:
    st.write("Data empty")


