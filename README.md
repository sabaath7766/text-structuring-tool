# Text Structuring Tool

A tool for structuring unstructured text data using AI and NLP methods. This project provides two primary approaches for text analysis: one using TF-IDF and the other using OpenAI's GPT.

## Table of Contents

- [About](#about)
- [Installation](#installation)
- [Usage](#usage)
  - [TF-IDF Method](#tf-idf-method)
  - [GPT Method](#gpt-method)
- [Setting Up Your OpenAI API Key](#setting-up-your-openai-api-key)
- [License](#license)

## About

This project is designed to transform unstructured text data into structured formats. It supports two approaches: one based on TF-IDF for comparing text to a gold standard, and another using OpenAI's GPT for extracting titles and structuring data.

## Installation

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/sabaath7766/text-structuring-tool.git
    cd text-structuring-tool
    ```

2. **Create a Virtual Environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### TF-IDF Method

1. **Run the TF-IDF Script**:
    ```bash
    python src/main.py
    ```
    - **Change Dataset**: To run the script on a different set of profiles, edit the dataset on line 198 in `main.py`.
    - **Change Gold Standard**: Update the corresponding gold standard on line 199.
    - **Change Output Name**: Modify the output name on line 186 to your preferred name.
    - **Output**: The script generates an HTML file comparing the gold standard to the output, which can be found in the specified output path.

### GPT Method

1. **Set Up Your OpenAI API Key** (See [Setting Up Your OpenAI API Key](#setting-up-your-openai-api-key)).

2. **Run the Title Extractor**:
    ```bash
    python src/title_extractor.py
    ```
    - **Change Dataset**: The default dataset size is set to 1000 lines. To change this or use your own dataset, modify line 72 in `title_extractor.py`.
    - **Output**: The raw output from the API will be saved as `.txt` files in the `output_data` folder.

3. **Convert to JSON**:
    - Run the JSON parser to convert the `.txt` output to `.json` format:
      ```bash
      python src/JSON_parser.py
      ```
    - The `.json` file will be created in the `output_data` folder.

4. **Search JSON Data**:
    - Use `search_tool.py` to search for keywords in the JSON files:
      ```bash
      python src/search_tool.py
      ```
    - **Configuration**: Change the configuration lines starting at line 56 in `search_tool.py` to specify the keyword and limit the search to specific entities.

## Disclaimer

**Cost of Running the GPT Method**

Please be aware that using the GPT method in this project involves accessing the OpenAI API, which incurs costs. For example, processing a dataset of 1000 lines typically costs around €7. Ensure you have sufficient balance on your OpenAI account to cover these expenses. For more detailed pricing information, visit the [OpenAI Pricing Page](https://openai.com/pricing).


## Setting Up Your OpenAI API Key

To use the GPT method, you need an OpenAI API key.

**Update the `api_key.txt` File**:
-  Place your OpenAI API key in a file named `APIkey` in the GPT directory of your project. The file should contain only the API key.

## License

This project is licensed under the MIT License.
