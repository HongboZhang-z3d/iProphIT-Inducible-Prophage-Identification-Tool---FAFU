# iProphIT
A deep learning approach that identifies the inducible activity of prophages from their DNA sequences.
## Requirements

System and software requirements:

- **Linux**
- **Python 3.x**    *(Any Python version compatible with PyTorch，Tested with Python 3.12.)*
- **biopython**
- **numpy**
- **pytorch**    *(If you want to enable GPU acceleration, please install the appropriate GPU-enabled PyTorch version from the official PyTorch website.)*

## Installation
**1.** You only need to download **`iProphIT-classifier.py`** and **`iProphIT_model-v1.pth`** into your working directory.  
(iProphIT_model-v1.pth website: (https://doi.org/10.5281/zenodo.17605580)   

**2.** Create a conda environment and install required packages:

```bash
conda create -n iprophit python=3.12
conda activate iprophit
conda install -c conda-forge biopython numpy
conda install pytorch
```

## Run iProphIT
**1.** Download **`iProphIT-classifier.py`**， **`iProphIT_model-v1.pth`** and put them in your working path.   

**2.** Run **`iProphIT-classifier.py`**

```bash
python iProphIT-classifier.py -i test_iProphIT.fasta -m iProphIT_model-v1.pth -o ./Result.tsv -t 16
```

## Usage

```bash
usage: iProphIT-classifier.py [-h] -i INPUT [-m MODEL] [-o OUTPUT] [-t THREADS]

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Path to the input FASTA file (required)
  -m MODEL, --model MODEL
                        Path to the trained model file (default: ./iProphIT_model-v1.pth)
  -o OUTPUT, --output OUTPUT
                        Output TSV file path (default: ./Result.tsv)
  -t THREADS, --threads THREADS
                        Number of CPU threads for DataLoader (default: 4)
```

## Typical output
- Find result in Result.tsv  

  | ID | Predict |
  |----------|:--------:|
  | prophage1     | active  |
  | prophage2  | dormant   |

- Explanation  
1.**`ID`** is the content of the description line in the genome file.   
2.**`Predict`** is the result of identification (**`active`**->inducible prophage, **`dormant`**->non-inducible prophage).

## Using testing data
genome file: **`OY731326.1`** and **`OY731419.1`**,   
source: Dahlman S. et al., Nature (2025), https://doi.org/10.1038/s41586-025-09614-7  
  
- Run **`iProphIT-classifier.py`**

```bash
python iProphIT-classifier.py -i test_iProphIT.fasta -m iProphIT_model-v1.pth -o ./Result.tsv -t 16
```

- Output of the test
```bash
ID	Predict
OY731326.1	active
OY731419.1	active
```

## Notes
- Input can accept genome files in formats such as `.fasta`, `.fa`, `.fna`, etc.
- iProphIT will automatically use the GPU if available, as long as you have installed a `PyTorch` version with CUDA support. 
## Copyright
Hongbo Zhang, Chen Liu, Hanpeng Liao, Fujian Provincial Key Laboratory of Soil Environmental Health and Regulation, College of Resources and Environment, Fujian Agriculture and Forestry University, Fuzhou, 350002, China.
