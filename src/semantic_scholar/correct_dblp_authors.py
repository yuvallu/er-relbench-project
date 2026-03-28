import csv
import html
import re
import json

def correct_json_syntax(s):
    if 'Alshrouf' in s:
        pass
    return re.sub(r"'", r'"', re.sub(r"Ngang'", r'###TEMP###', s)).replace("None", "null").replace('###TEMP###', "Ngang'")


# Function to properly format array fields for PostgreSQL, accounting for semicolon delimiters and quotes
def correct_and_format_array_field(field):
    if not field:
        return '{}'
    
    # Decode HTML entities and split elements on semicolons
    elements = [html.unescape(elem.strip()) for elem in field.split(';') if elem]
    
    # Correctly escape quotes and wrap elements with double quotes for PostgreSQL array syntax
    corrected_elements = ['"{}"'.format(elem.replace('"', '""')) for elem in elements]
    
    # Join elements into a PostgreSQL array representation
    corrected_array = '{' + ','.join(corrected_elements) + '}'
    return corrected_array

# Adjusted script to read and correct CSV data for the authors file
input_csv_path = 'authors.csv'
output_csv_path = '/etc/postgresql/16/main/authors_corrected.csv'

with open(input_csv_path, 'r', encoding='utf-8') as infile, open(output_csv_path, 'w', encoding='utf-8', newline='') as outfile:
    reader = csv.DictReader(infile)
    writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
    
    # Write the header
    writer.writeheader()
    
    for row in reader:
        # Correct and format the aliases field
        row['aliases'] = correct_and_format_array_field(row['aliases'])
        # Correct and format the affiliations field
        row['affiliations'] = correct_and_format_array_field(row['affiliations'])
        
        # Correct the JSON formatting for the externalids column
        row['externalids'] = correct_json_syntax(row['externalids'])
        
        writer.writerow(row)

print(f"Processed file saved as {output_csv_path}")
