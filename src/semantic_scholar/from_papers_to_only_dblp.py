import csv
import ast

# Define input and output CSV file paths
input_csv_path = 'papers.csv'
output_csv_path = 'dblp_papers.csv'

# Open the input CSV file for reading and the output CSV file for writing
with open(input_csv_path, mode='r', encoding='utf-8') as infile, \
        open(output_csv_path, mode='w', newline='', encoding='utf-8') as outfile:
    # Create CSV reader and writer objects
    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)

    # Write the header to the output CSV file
    writer.writeheader()

    # Iterate over each row in the input CSV file
    for row in reader:
        try:
            # Convert the 'externalids' string to a dictionary
            external_ids_dict = ast.literal_eval(row['externalids'])

            # Check if the 'DBLP' key is not None
            if external_ids_dict.get('DBLP') is not None:
                # Write the row to the output CSV file
                writer.writerow(row)
        except ValueError as e:
            # Handle possible errors during string to dictionary conversion
            print(f"Error processing row: {e}")

print("Filtering complete. Rows with 'DBLP' not None have been written to the output CSV file.")
