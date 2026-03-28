import csv
import html
import re

# Function to correct JSON formatting for the externalids column
def correct_json_syntax(s):
    return re.sub(r"'", r'"', re.sub(r"Ngang'", r'###TEMP###', s)).replace("None", "null").replace('###TEMP###', "Ngang'")

# Function to properly format array fields for PostgreSQL, accounting for semicolon delimiters and quotes
def correct_and_format_array_field(field):
    field_without_quotation_mark = field.replace('\\',"").replace('"','')
    # Decode HTML entities
    decoded_field = html.unescape(field_without_quotation_mark)
    
    # Split elements on semicolons instead of commas for specific fields like authors
    # This approach also handles the case of trailing semicolons by filtering out empty strings
    elements = [elem.strip() for elem in decoded_field.strip('{}').split(';') if elem]
    
    # Properly escape and quote elements
    corrected_elements = []
    for element in elements:
        # Double existing quotes for proper escaping
        element = element.replace('"', '""')
        
        # Enclose each element in double quotes
        corrected_elements.append(f'"{element}"')
    
    # Reassemble into a properly formatted array literal
    corrected_array = '{' + ','.join(corrected_elements) + '}'
    return corrected_array

# Adjusted script to read and correct CSV data
input_csv_path = '/etc/postgresql/16/main/dblp_papers.csv'
output_csv_path = '/etc/postgresql/16/main/dblp_papers_corrected.csv'

with open(input_csv_path, 'r', encoding='utf-8') as infile, open(output_csv_path, 'w', encoding='utf-8', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    # Copy the header without modification
    writer.writerow(next(reader))
    
    for row in reader:
        # Correct and format the authors, s2fieldsofstudy, and publicationtypes fields
        row[4] = correct_and_format_array_field(row[4])  # Correct authors column
        row[12] = correct_and_format_array_field(row[12])  # Correct s2fieldsofstudy column
        row[-3] = correct_and_format_array_field(row[-3])  # Correct publicationtypes column
        
        # Correct the JSON formatting for the externalids column
        row[1] = correct_json_syntax(row[1])  # Correct externalids column
        
        writer.writerow(row)
