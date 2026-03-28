import json
import csv
import gzip
import os

# Define the base path for the JSON and CSV files
base_json_path = '/tmp/authors_'
base_csv_path = './authors.csv'

# Updated CSV headers for author data
headers = [
    "authorid",
    "externalids",
    "url",
    "name",
    "aliases",
    "affiliations",
    "homepage",
    "papercount",
    "citationcount",
    "hindex"
]

# Helper function to format nested fields
def format_field(data, key):
    value = data.get(key)
    if value is None:  # If the value is None, return an empty string
        return ''

    if key in ['aliases', 'affiliations']:  # Assuming affiliations might need similar handling
        # Join list values into a string; modify as needed based on actual structure
        return '; '.join(value) if isinstance(value, list) else ''
    else:
        return str(value)

# Update the process_file function to handle exceptions and modified structure
def process_file(json_file_path, csv_file_path, mode='a'):
    with gzip.open(json_file_path, 'rt', encoding='utf-8') as json_file, \
         open(csv_file_path, mode, newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=headers)
        if mode == 'w':  # Write header only once
            csv_writer.writeheader()

        for line in json_file:
            data = json.loads(line)
            try:
                csv_data = {header: format_field(data, header) for header in headers}
                csv_writer.writerow(csv_data)
            except Exception as e:
                print(f"An error occurred: {e}")
                continue  # Skip this line and continue with the next

# Process files
for i in range(61):  # Adjust the range as needed based on the number of files
    print(f"We are at file number - {i}")
    file_suffix = str(i).zfill(3)
    json_file_path = f'{base_json_path}{file_suffix}.gz'

    # Check if the .gz file exists
    if os.path.exists(json_file_path):
        if i == 0:
            process_file(json_file_path, base_csv_path, 'w')  # 'w' to write the header in the first file
        else:
            process_file(json_file_path, base_csv_path, 'a')  # 'a' to append subsequent files

print(f"CSV file has been created and appended at {base_csv_path}")
