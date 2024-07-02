import tarfile
import os

tar_path = r"E:\Projects\MusicBench_Data\MusicBench.tar.gz"
extract_path = r"E:\Projects\MusicBench_Data\extracted_files"

# Untar and extract the .tar.gz files
if __name__ == "__main__":
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)

    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(extract_path)
    except:
        print("Error extracting the files.")
