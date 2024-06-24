import json
import re
import ast

def extract_and_compile_json(input_file_path, output_file_path):
    json_pattern = re.compile(r'\{.*?\}', re.DOTALL)
    all_json_objects = []

    with open(input_file_path, 'r', encoding='utf-8') as file:
        file_content = file.read()

    match_index = 0
    for match in json_pattern.finditer(file_content):
        json_str = match.group()
        try:
            # Parse the JSON string using ast.literal_eval as a fallback
            json_obj = ast.literal_eval(json_str)
            if not isinstance(json_obj, dict):
                raise ValueError("Parsed object is not a dictionary.")

            # Perform replacements in the values of the dictionary
            json_obj = {k: post_process_value(v) for k, v in json_obj.items() if v != ''}
            json_obj['index'] = match_index
            all_json_objects.append(json_obj)
            match_index += 1
        except (ValueError, SyntaxError) as e:
            error_msg = f"Error processing JSON-like data at match index {match_index}: {e}"
            print(error_msg)
            match_index += 1

    # Save the processed and post-processed JSON objects to a file
    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        json.dump(all_json_objects, outfile, indent=4)

    print(f"Processed and saved {len(all_json_objects)} JSON objects into '{output_file_path}'.")

def post_process_value(value):
    if isinstance(value, str):
        # Replace the right single quotation mark with an apostrophe
        return value.replace('\u2019', "'")
    return value


# Specify your input and output file paths
input_file_path = 'output_data/output_20240309_130328-BigOne.txt'
output_file_path = 'output_data/output_20240309_130328-BigOne.json'

# Run the function
extract_and_compile_json(input_file_path, output_file_path)
