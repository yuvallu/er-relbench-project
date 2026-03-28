import json
import csv
import gzip
import os

# Define the base path for the JSON and CSV files
base_json_path = '/tmp/papers_'
base_csv_path = './papers.csv'

# Extend CSV headers to include new keys
headers = [
    "corpusid",
    "externalids",
    "url",
    "title",
    "authors",
    "venue",
    "publicationvenueid",
    "year",
    "referencecount",
    "citationcount",
    "influentialcitationcount",
    "isopenaccess",
    "s2fieldsofstudy",
    "publicationtypes",
    "publicationdate",
    "journal"
]


# Helper function to format nested fields
def format_field(data, key):
    value = data.get(key)
    if value is None:  # If the value is None, return an empty string
        return ''

    if key == 'authors':
        return '; '.join([author['name'] for author in value])
    elif key == 's2fieldsofstudy':
        return '; '.join([fos['category'] for fos in value])
    elif key == 'journal':
        return value.get('name', '') if isinstance(value, dict) else ''
    else:
        return str(value)

# Update the process_file function to handle exceptions
def process_file(json_file_path, csv_file_path, mode='a'):
    with open(json_file_path, 'r') as json_file, open(csv_file_path, mode, newline='', encoding='utf-8') as csv_file:
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


# Process files from papers_000 to papers_060
for i in range(61):  # 61 because range is exclusive on the end value
    print(f"We are at file number - {i}")
    file_suffix = str(i).zfill(3)
    json_file_path = f'{base_json_path}{file_suffix}.gz'

    # Check if the .gz file exists
    if os.path.exists(json_file_path):
        # Decompress .gz file to temporary file
        with gzip.open(json_file_path, 'rt', encoding='utf-8') as compressed_file:
            temp_json_path = f'/tmp/temp_papers_{file_suffix}.json'
            with open(temp_json_path, 'w', encoding='utf-8') as temp_file:
                temp_file.write(compressed_file.read())

        # Now process this temporary JSON file
        if i == 0:
            process_file(temp_json_path, base_csv_path, 'w')  # 'w' to write the header in the first file
        else:
            process_file(temp_json_path, base_csv_path, 'a')  # 'a' to append subsequent files

        # Cleanup: remove the temporary JSON file
        os.remove(temp_json_path)

print(f"CSV file has been created and appended at {base_csv_path}")








# import json
# import csv
# import gzip
# import os
#
# # Define the base path for the JSON and CSV files
# base_json_path = '/tmp/papers_'
# base_csv_path = '/tmp/papers.csv'
#
# # Define the CSV headers
# headers = ['corpusid', 'title', 'year', 'referencecount', 'citationcount', 'influentialcitationcount', 'isopenaccess', 'authors']
#
#
# # Function to process each JSON file and append data to a CSV file
# def process_file(json_file_path, csv_file_path, mode='a'):
#     with open(json_file_path, 'r') as json_file, open(csv_file_path, mode, newline='', encoding='utf-8') as csv_file:
#         csv_writer = csv.DictWriter(csv_file, fieldnames=headers)
#         if mode == 'w':  # Write header only once
#             csv_writer.writeheader()
#
#         for line in json_file:
#             data = json.loads(line)
#             csv_data = {
#                 'corpusid': data.get('corpusid'),
#                 'title': data.get('title'),
#                 'year': data.get('year'),
#                 'referencecount': data.get('referencecount'),
#                 'citationcount': data.get('citationcount'),
#                 'influentialcitationcount': data.get('influentialcitationcount'),
#                 'isopenaccess': data.get('isopenaccess'),
#                 'authors': '; '.join([author['name'] for author in data.get('authors', [])])
#             }
#             csv_writer.writerow(csv_data)
#
# # Process files from papers_000 to papers_060
# for i in range(61):  # 61 because range is exclusive on the end value
#     file_suffix = str(i).zfill(3)
#     json_file_path = f'{base_json_path}{file_suffix}.gz'
#
#     # Check if the .gz file exists
#     if os.path.exists(json_file_path):
#         # Decompress .gz file to temporary file
#         with gzip.open(json_file_path, 'rt', encoding='utf-8') as compressed_file:
#             temp_json_path = f'/tmp/temp_papers_{file_suffix}.json'
#             with open(temp_json_path, 'w', encoding='utf-8') as temp_file:
#                 temp_file.write(compressed_file.read())
#
#         # Now process this temporary JSON file
#         if i == 0:
#             process_file(temp_json_path, base_csv_path, 'w')  # 'w' to write the header in the first file
#         else:
#             process_file(temp_json_path, base_csv_path, 'a')  # 'a' to append subsequent files
#
#         # Cleanup: remove the temporary JSON file
#         os.remove(temp_json_path)
#
# print(f"CSV file has been created and appended at {base_csv_path}")
