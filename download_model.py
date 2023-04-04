import os
from google.cloud import storage
from pathlib import Path
from tqdm import tqdm

bucket_name = 'public-kenricklancebunag'
prefix = 'checkpoint-263820/'
dl_dir = 'project/api/ML'

storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)

blobs = bucket.list_blobs(prefix=prefix)  # Get list of files
num_files = len(list(blobs))
progress_bar = tqdm(range(num_files))

blobs = bucket.list_blobs(prefix=prefix)
for blob in blobs:
    if blob.name.endswith("/"):
        continue
    progress_bar.set_description(f'Downloading {blob.name}')

    file_split = blob.name.split("/")
    directory = "/".join(file_split[0:-1])
    local_directory = os.path.join(dl_dir, directory)
    Path(local_directory).mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(os.path.join(dl_dir, blob.name))

    progress_bar.update(1)