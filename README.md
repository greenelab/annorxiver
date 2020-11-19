# Annorxiver

## Citation
- [Manuscript in progress](https:/greenelab.github.io/annorxiver_manuscript)
- doi and full citation coming soon!!

## About
This repository contains code for the annorxiver project.
This project analyzes the linguistic style and content of bioRxiv preprints and aims to understand how these features change when preprints undergo the publication process.
We are currently working on a manuscript for the work in this repository.
Feel free to check it out! (link above).
We also created a [web app](greenelab.github.io/preprint-similarity-search) that takes a bioRxiv or medRxiv preprint doi as input and returns a set of the most linguistically similar journals and articles to serve as potential publication venues for their work.

## Directory Structure
| Folder/file | Description |
| --- | --- | 
| [annorxiver_modules](annorxiver_modules) | This folder contains supporting functions that other notebooks in this repository will use |
| [biorxiv](biorxiv) | This folder contains all experiments that are related to biorxiv preprints. | 
| [credentials](credentials) | Some of the code in this repository need credentials to run. Any user that needs to run those notebooks should check this folder first. |
| [figure_generation](figure_generation) | This folder contains code to generate figures for the manuscript in progress. |
| [nytac/corpora_stats](nytac/corpora_stats) | This folder contains results when parsing the New York Times annotated corpus (NYTAC) from the Linguistic Data Consortium (LDC). |
| [pmc](pmc) |  This folder contains all experiments that are related to articles in Pubmed Central Open Access corpus (PMC). | 
| environment.yml | This file contains the necessary packages this repository uses.  |
| setup.py | This file sets up the annorxiver modules to be used as a regular python package. |


## Set up Environment

Annorxiver uses [conda](http://conda.pydata.org/docs/intro.html) as a python package manager. Before moving on to the instructions below, please make sure to have it installed. [Download conda here!!](https://docs.conda.io/en/latest/miniconda.html)
  
Once everything has been installed, type following command in the terminal: 

```bash
conda env create --file environment.yml
``` 

You can activate the environment by using the following command: 

```bash
source activate annorxiver
```  

_Note_: If you want to leave the environment, just enter the following command:

```bash
source deactivate 
```

## License

This repository is dual licensed as [BSD 3-Clause](LICENSE-BSD.md) and [CC0 1.0](LICENSE-CC0.md), meaning any repository content can be used under either license. This licensing arrangement ensures source code is available under an [OSI-approved License](https://opensource.org/licenses/alphabetical), while non-code content — such as figures, data, and documentation — is maximally reusable under a public domain dedication.
