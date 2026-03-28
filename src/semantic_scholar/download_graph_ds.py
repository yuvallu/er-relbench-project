import requests


def download_dataset(url, file_path, mode='wb'):
    with requests.get(url, stream=True) as response:
        response.raise_for_status()  # Check for errors
        with open(file_path, mode) as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)


# Define base URL for datasets API
base_url = "https://api.semanticscholar.org/datasets/v1/release/"

# This endpoint requires authentication via api key
api_key = "XoU4c9tb0h28N7aHzbQGJ9InNaAlCG2y7AOpecPB"
headers = {"x-api-key": api_key}

# Make the initial request to get the list of releases
response = requests.get(base_url)

if response.status_code == 200:
    # Assume we want data from the latest release, which will correspond to the last item in the response list since releases are ordered chronologically
    release_id = response.json()[-1]

    # Make a request to get datasets available in the latest release (this endpoint url is the release id appended to the base url)
    datasets_response = requests.get(base_url + release_id)

    if datasets_response.status_code == 200:
        # Fetch the datasets list from the response
        datasets = datasets_response.json()['datasets']

        # Check if the 'papers' dataset exists
        papers_dataset_exists = any(dataset.get('name') == 'papers' for dataset in datasets)

        if papers_dataset_exists:
            # Make a request to get download links for the 'papers' dataset
            dataset_name = 'papers'
            download_links_response = requests.get(base_url + release_id + '/dataset/' + dataset_name, headers=headers)

            if download_links_response.status_code == 200:
                download_links = download_links_response.json()["files"]

                # Your code to process the download links goes here
                print(download_links)
                file_path = './mongo_graph.json'
                for index, link in enumerate(download_links):
                   print(f"Downloading link number {index}")
                   # If index is 0, it's the first iteration so mode is 'wb'. Otherwise, it's 'ab'.
                   mode = 'wb' if index == 0 else 'ab'
                   download_dataset(link, file_path, mode=mode)

            else:
                print(f"Failed to get download links. Status code: {download_links_response.status_code}")
        else:
            print("The 'papers' dataset does not exist in the list.")
    else:
        print(f"Failed to get datasets. Status code: {datasets_response.status_code}")
else:
    print(f"Failed to get releases. Status code: {response.status_code}")