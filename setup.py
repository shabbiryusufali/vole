import fasttext
import fasttext.util
import os
import urllib.request
import zipfile

script_path = os.path.abspath(__file__)
base_directory = os.path.dirname(script_path)


def setup_fasttext_model():
    fasttext.util.download_model("en", if_exists="ignore")
    print("Loading FastText model...")
    ft = fasttext.load_model("cc.en.300.bin")
    print("Reducing FastText model...")
    fasttext.util.reduce_model(ft, 100)
    print("Saving FastText model...")
    ft.save_model(base_directory + "/vole/models/cc.en.100.bin")
    print("Cleaning up FastText files...")
    os.remove("cc.en.300.bin")
    os.remove("cc.en.300.bin.gz")


def setup_SARD_dataset():
    if not os.path.exists(base_directory + "/data"):
        print('Creating "data" directory...')
        os.makedirs(base_directory + "/data")

    directory = os.path.join(base_directory, "data")
    print(f"Setting up SARD dataset in {directory}...")
    url = "https://samate.nist.gov/SARD/downloads/test-suites/2017-10-01-juliet-test-suite-for-c-cplusplus-v1-3.zip"

    zip_path = os.path.join(directory, "juliet-test-suite.zip")
    print(f"Downloading SARD dataset from {url}...")
    urllib.request.urlretrieve(url, zip_path)

    print("Extracting SARD dataset...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(directory)

    os.remove(zip_path)
    os.rename(directory + "/C", directory + "/SARD")
    print("SARD Dataset Ready!")


def main():
    setup_fasttext_model()
    setup_SARD_dataset()
    print("Setup complete!")


if __name__ == "__main__":
    main()
