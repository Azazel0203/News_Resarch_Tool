# News Research Tool

## Overview
The News Research Tool is a Streamlit-based application that allows users to input URLs of news articles, process the URLs to extract relevant information, and save the extracted data to a FAISS index. Users can then input questions and retrieve answers from the indexed data using a language model. The application utilizes the LangChain library, OpenAI, and FAISS for various natural language processing tasks.

## Getting Started
1. Clone this repository.
2. Install the required dependencies using the following command:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a file named `.env` in the project directory and set your OpenAI API key as follows:
   ```plaintext
   OPENAI_API_KEY=your_openai_api_key_here
   ```
4. Run the application:
   ```bash
   streamlit run main.py
   ```

## Usage
1. Enter up to three news article URLs in the sidebar.
2. Click the "Process URLs" button to load, split, embed, and save the data to a FAISS index.
3. Enter a question in the main input box.
4. View the answer and sources, if available, in the Streamlit app.

## Code Structure
- `main.py`: Contains the Streamlit application code.
- `langchain_community/`: External module for LangChain community additions.
- `langchain_openai.py`: OpenAI embeddings module.
- `faiss_store_openai.pkl`: Pickle file containing the FAISS index.

## Custom Functions
### `get_result(URLs, main_placeholder, file_path)`
This function takes a list of URLs, a placeholder for displaying status messages, and a file path. It loads data from the provided URLs, splits the text into chunks, creates embeddings using OpenAI, and saves the embeddings to a FAISS index stored in a pickle file.

```python
# Example Usage
get_result(["url1", "url2", "url3"], main_placeholder, "faiss_store_openai.pkl")
```

## Dependencies
- `streamlit`: Web application framework.
- `langchain`: Natural language processing library.
- `dotenv`: Load environment variables.
- `pickle`: Serialization module for saving the FAISS index.
- `time`: Module for introducing delays in the process.

## Note
Make sure to replace `"your_openai_api_key_here"` in the `.env` file with your actual OpenAI API key for the tool to function correctly.
