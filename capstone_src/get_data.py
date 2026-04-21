import argparse
import subprocess
from pathlib import Path
import requests
import tarfile
from tqdm import tqdm

from datetime import datetime
import logging



def extract_tar(archive_path: Path, extract_to: Path, gz=True, remove_archive=False):

    logging.info(f"Extracting {archive_path}")
    extract_to.mkdir(parents=True, exist_ok=True)

    try:
        if gz:
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(path=extract_to, filter="data")
        else:
            with tarfile.open(archive_path, "r") as tar:
                tar.extractall(path=extract_to, filter="data")
    except Exception as e:
        logging.info(f"Error extracting {archive_path}: {e}")
            
    if remove_archive:
        try:
            archive_path.unlink()
            logging.info(f"Removed archive {archive_path}")
        except Exception as e:
            logging.info(f"Could not remove archive {archive_path}: {e}")
        

def download_file(url: str, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        logging.info(f"File already exists: {output_path}")
        return

    logging.info(f"Downloading {url}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("Content-Length", 0))
        chunk_size = 8192

        with open(output_path, "wb") as f, tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=f"Downloading {output_path.name}",
        ) as pbar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:  # filter out keep-alive chunks
                    f.write(chunk)
                    pbar.update(len(chunk))
                

def download_tar_dataset(url: str, download_path, name: str, gz=True):
    raw_dir = Path(download_path)
    
    if (raw_dir / name).exists():
        logging.info(f"{name} dataset already downloaded and extracted at {raw_dir / name}")
        return
    
    archive_path = raw_dir / (f"{name}.tar.gz" if gz else f"{name}.tar")
    extract_dir = raw_dir

    download_file(url, archive_path)
    extract_tar(archive_path, extract_dir, gz)



def clone_styletalk(path):
    """
    Downloads StyleTalk files from Google Drive, clones repo for annotations
    
    https://github.com/DanielLin94144/StyleTalk
    """
    data_dir = Path(path)
    annot_dir = data_dir / "annotations"
    # Cloning annotations
    if annot_dir.exists():
        logging.info(f"StyleTalk annotations already exists at {annot_dir}")
        
    else:
        # LibriTTS-P (extra annotation data)
        subprocess.run(
            [
                "git", "clone",
                "https://github.com/DanielLin94144/StyleTalk.git",
                str(annot_dir),
            ],
            check=True,
        )
        
    # Downloading audio
    

    url = "https://drive.google.com/uc?id=12mlGaZkkBebk4gf0qY684W_bwvmCsu_t"
    name = "audio"
    archive_path = data_dir / f"{name}.tar.gz"
    
    extract_dir = data_dir
    tar_top_folder = "audio" # this is within the tar, so i pass just extract_dir to get correct structure
    final_structure = extract_dir / tar_top_folder 
    
    if final_structure.exists():
        logging.info(f"StyleTalk audio already exists at {final_structure}")
    elif  archive_path.exists():
        logging.info(f"StyleTalk audio tar already exists at {archive_path}")
    else:
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        
        import gdown
        gdown.download(url,  str(archive_path), quiet=False)

    # extract
    
    if final_structure.exists():
        logging.info(f"StyleTalk audio already extracted at {final_structure}")
    else:
        extract_tar(archive_path, extract_dir)
        
    
    
def download_ears(foldername: Path):
    """Recreated from EARS github repository download_ears.py"""
    import os
    import urllib.request
    import zipfile
    
    def download_file(url, filename):
        with urllib.request.urlopen(url) as response:
            data = response.read()
            with open(filename, 'wb') as file:
                file.write(data)


    def unzip_file(zip_path, extract_to):
        os.makedirs(extract_to, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

    # these speaker ids were chosen to EXCLUDE
    # because they have ethnicity="white/caucasian" and native language="american english",
    # i.e. the majority values in the EARS dataset. exlcuding these will help balance the dataset and 
    # reduce the number of files to download (87 speakers remaining, ~ 87 hours)
    
    exclude_these_speakers = ['p034','p002','p076','p011','p084','p028','p019','p051','p066','p022',
                              'p049','p018','p102','p014','p093','p048','p088','p054','p087','p070']
    
    foldername.mkdir(parents=True, exist_ok=True)
    (foldername / "zips").mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(1, 108), desc=f"download {107 - len(exclude_these_speakers)} speakers of EARS dataset (skipping {len(exclude_these_speakers)} speakers)"):
        if f"p{i:03d}" in exclude_these_speakers:
            continue
        url = f"https://github.com/facebookresearch/ears_dataset/releases/download/dataset/p{i:03d}.zip"
        zip_file = str(foldername / "zips_temp" / f"p{i:03d}.zip")
        wav_parent = str(foldername)
        download_file(url, zip_file)
        unzip_file(zip_file, wav_parent )
        os.remove(zip_file)
        
    os.remove(str(foldername / "zips_temp"))
    
    
def download_emilia(parent_Path: Path, hf_token):
    from huggingface_hub import hf_hub_download
    
    tar_subset = ["EN_B00000","EN_B00001","EN_B00011","EN_B00012"
                  ,"EN_B00017","EN_B00019","EN_B00030","EN_B00033"
                  ,"EN_B00037","EN_B00041","EN_B00044","EN_B00047"
                ]
    
    
    parent_Path.mkdir(parents=True, exist_ok=True)
    
    final_path = parent_Path / "EN"
    
    logging.info(f"Emilia: Downloading {len(tar_subset)} big tars. This will take hours.")
    for filename in tar_subset:
        
        extracted_Path = final_path / filename
        tar_file_Path = final_path / Path(str(filename) + ".tar.gz")
        
        # check if this tar is already extracted
        download = True
        extract = True
        if extracted_Path.exists():
            logging.info(f"{str(extracted_Path)} already exists. skipping download & extract.")
            download = False
            extract = False
        elif tar_file_Path.exists():
            logging.info(f"{str(tar_file_Path)} already exists. skipping to extract.")
            download = False
        
        if download:
            local_path = hf_hub_download(
                repo_id="amphion/Emilia-Dataset",
                filename=f"EN/{filename}.tar.gz",
                repo_type="dataset",
                token=hf_token,
                revision="fc71e07e8572f5f3be1dbd02ed3172a4d298f152",
                local_dir=parent_Path      
            )
        
        if extract:
            extract_tar(Path(local_path), final_path, remove_archive=True)
    


def preprocess_expresso(expresso_root):
    def execute(cmd, cwd=None):
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=cwd)
        
        for line in process.stdout:
            print(line, end="")
        
        # wait for process to finish and grab stderr
        _, stderr = process.communicate()
        if stderr:
            print("STDERR:", stderr)
        if process.returncode != 0:
            raise RuntimeError(f"Command failed with exit code {process.returncode}: {cmd}")

    execute(["python", "data_helpers/expresso_vad_multi.py", expresso_root])

    execute(["python", "data_helpers/normalize.py"])
    
        
def clone_paraspeechcaps(path, hf_token, include_all_audio=False):
    """Download helper scripts, annotations files, and select audio files for ParaSpeechCaps dataset.
    
    **Note**: Does not download all audio for ParaSpeechCaps, only the audio used for this project:
    
        1. Expresso. Also completes preprocessing steps suggested in the ParaSpeechCaps repo
    
    Instructions:
    https://github.com/ajd12342/paraspeechcaps?tab=readme-ov-file#2-paraspeechcaps-dataset 
    """
    
    from datasets import load_dataset, load_dataset_builder
    import os
    
    data_dir = Path(path)
    annot_dir = data_dir / "annotations"
    # Cloning annotations
    if annot_dir.exists():
        logging.info(f"ParaSpeechCaps repo already exists at {annot_dir}")
   
        
    else:
        subprocess.run(
            [
                "git", "clone",
                "https://github.com/ajd12342/paraspeechcaps.git",
                str(annot_dir)
            ],
            check=True,
        )
    
    psc_dataset_name = "ajd12342/paraspeechcaps"
    logging.info("Checking if ParaSpeechCaps already cached.")
    builder = load_dataset_builder(psc_dataset_name)
    # this is the directory HF would use for the processed dataset
    cache_dir = builder.cache_dir

    if os.path.exists(cache_dir):
        logging.info("ParaSpeechCaps already cached. SKIPPING.")
    else:
        # Download and cache the dataset
        logging.info("Not cached. Downloadign ParaSpeechCaps")
        _ = load_dataset("ajd12342/paraspeechcaps")

        
    # Audio datasets
    
    data_dir_audio = data_dir / "audio"
        
    # Download Expresso dataset
    logging.info("Downloading audio from EXPRESSO dataset")
    expresso_url = "https://dl.fbaipublicfiles.com/textless_nlp/expresso/data/expresso.tar"
    name = "expresso"
    download_tar_dataset(expresso_url, data_dir_audio, name, gz=False)

    expresso_root = data_dir_audio / name
    preprocess_expresso(str(expresso_root))
    
    if include_all_audio:
        # Download ears
        ears_audio = data_dir_audio / "EARS"
        if ears_audio.exists():
            logging.info("EARS audio folder already exists")
        else:
            download_ears(ears_audio)
            
        # Download subset of Emilia
        emilia_audio_path = data_dir_audio / "Emilia"
        download_emilia(emilia_audio_path, hf_token)


def main():
    parser = argparse.ArgumentParser(
        description="Download datasets required for this project."
    )

    parser.add_argument(
        "--styletalk",
        action="store_true",
        help="Download the styletalk dataset",
    )
    
    parser.add_argument(
        "--paraspeechcaps",
        action="store_true",
        help="Download the paraspeechcaps dataset",
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all dataset",
    )
    
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Include this if also passing --paraspeechcaps or --all. Hugging Face access token for private dataset downloads",
    )

    args = parser.parse_args()
    
    
    
    # init logging
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    log_path = Path("logs")
    log_path.mkdir(exist_ok=True)
    log_filename = f"{str(log_path)}/get_data_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()  # <-- this is console
        ]
    )

    if not (args.styletalk or args.paraspeechcaps or args.all):
        parser.error("Please specify at least one dataset to download.")

    elif args.styletalk:
        clone_styletalk("data/raw/styletalk")
        
    elif args.paraspeechcaps:
        clone_paraspeechcaps("data/raw/paraspeechcaps", args.hf_token)
        
    elif args.all:
        clone_styletalk("data/raw/styletalk")
        clone_paraspeechcaps("data/raw/paraspeechcaps", args.hf_token)


if __name__ == "__main__":
    main()
