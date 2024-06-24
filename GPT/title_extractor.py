from openai import OpenAI
import os
from datetime import datetime
from tqdm import tqdm
import time

# Setup OpenAI API
with open("APIkey", "r") as file:
    api_key = file.read().strip()

client = OpenAI(api_key=api_key)


def extract_features(text):
    start_time = time.time()
    lines = text.split('\n')  # Splitting the text into lines
    extracted_features = ""
    total_lines = len(lines)

    for line in tqdm(lines, total=total_lines, desc="Processing"):
        if line.strip():  # Skip empty lines
            try:
                response = client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a data extractor and formatter. Your task is to convert news headlines "
                                       "into a structured key-value format suitable for a document database. You "
                                       "should aim for minimal columns and concise content, ensuring each cell has "
                                       "limited characters but still preserves the essential information of the "
                                       "headline. Exclude redundant details. Focus on extracting and structuring the "
                                       "essential elements from each headline into the following format. {\"Entity\": \"\","
                                       " \"Action\": \"\", \"Location\": \"\", \"Time\": \"\", \"Impact\": \"\", "
                                       "\"Reason\": \"\"}"
                        },
                        {
                            "role": "user",
                            "content": f"Process this headline: \n{line}"
                        }
                    ],
                    model="gpt-4-0125-preview",
                    seed=5852
                    # max_tokens=100
                )

                features_text = response.choices[0].message.content
                extracted_features += features_text + "\n"
            except Exception as e:
                print(f"Error processing line: {e}")
                continue

    end_time = time.time()  # End timing
    print(f"Feature extraction completed in {end_time - start_time:.2f} seconds.")

    return extracted_features.strip()


def save_output(output, folder="output_data"):
    # Ensure the output directory exists
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Format the filename with the current datetime
    filename = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    # Save the output to a file
    with open(os.path.join(folder, filename), 'w') as file:
        file.write(output)


def main():
    with open("dataset/mix-1000", "r") as file:
        file_text = file.read()

    all_features = extract_features(file_text)
    print(all_features)

    # Save the output
    if all_features is not None:
        save_output(all_features)


if __name__ == "__main__":
    main()
