import json
import time

def load_json(file_path):
    """Load JSON data from a file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def search_keyword_in_documents(documents_data, keyword, *search_fields):
    matching_indexes = []
    for doc in documents_data:
        for field in search_fields:
            if field in doc and isinstance(doc[field], str) and keyword.lower() in doc[field].lower():
                matching_indexes.append(doc['index'])
                break  # Found a match in one of the specified fields
    return matching_indexes

def find_text_by_index(text_data, index):
    for doc in text_data:
        if doc['index'] == index:
            return doc['text']
    return "Text not found"

def search_and_display(keyword, titles_file_path, text_file_path, output_file_path, *search_fields):
    start_time = time.time()  # Start timing for JSON search

    titles_data = load_json(titles_file_path)
    text_data = load_json(text_file_path)

    matching_indexes = search_keyword_in_documents(titles_data, keyword, *search_fields)
    matching_texts = [find_text_by_index(text_data, index) for index in matching_indexes]

    # Write matching texts to an output file
    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        for text in matching_texts:
            outfile.write(text + '\n')

    end_time = time.time()  # End timing for JSON search
    print(f"Fields searched: {', '.join(search_fields)}")
    print(f"Total documents matched: {len(matching_indexes)}")
    print(f"JSON search completed in {end_time - start_time:.4f} seconds.")

def search_in_txt_file(file_path, keyword):
    start_time = time.time()  # Start timing for text file search

    count = 0
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if keyword.lower() in line.lower():
                count += 1

    end_time = time.time()  # End timing for text file search
    print(f"Total lines outputted: {count}")
    print(f"Text file search completed in {end_time - start_time:.4f} seconds.")

# Configuration
titles_file_path = 'output_data/output_20240309_130328-bigOne.json'
text_file_path = 'output_data/text-1000.json'
txt_file_path = 'dataset/title-text-1000'
output_file_path = 'output_data/matching_texts.json'
keyword = 'Trump'
search_fields = ['Entity']  # Adjust fields as needed

# Execute the function for JSON search and output
search_and_display(keyword, titles_file_path, text_file_path, output_file_path, *search_fields)

# Execute the function for text file search
search_in_txt_file(txt_file_path, keyword)
