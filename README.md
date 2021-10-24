# Annorxiver

## Citation
- BioRxiv Citation: [10.1101/2021.03.04.433874](https://doi.org/10.1101/2021.03.04.433874)
- Publication Citation: (TBD)

## About
This repository contains code for the annorxiver project.
This project analyzes the linguistic content and style of bioRxiv preprints and aims to understand how these features change when preprints undergo the publication process.
Our manuscript is currently finished and under review at the Plos Biology journal.
Feel free to check our manuscript out! (link above).
We also created a [web app](greenelab.github.io/preprint-similarity-search) that takes a bioRxiv or medRxiv preprint doi as input and returns a set of the most linguistically similar journals and articles to serve as potential publication venues for their work.

## Data Availability
Data for each figure in our manuscript can be found in our [FIGURE_DATA_SOURCE.md](FIGURE_DATA_SOURCE.md) file.
This file contains relative links for each data source used to generate each piece of the figure panel.

## Directory Structure
| Folder/file | Description |
| --- | --- | 
| [annorxiver_modules](annorxiver_modules) | This folder contains supporting functions that other notebooks in this repository will use |
| [biorxiv](biorxiv) | This folder contains all experiments that are related to biorxiv preprints. | 
| [credentials](credentials) | Some of the code in this repository need credentials to run. Any user that needs to run those notebooks should check this folder first. |
| [figure_generation](figure_generation) | This folder contains code to generate figures for the manuscript in progress. |
| [nytac/corpora_stats](nytac/corpora_stats) | This folder contains results when parsing the New York Times annotated corpus (NYTAC) from the Linguistic Data Consortium (LDC). |
| [pmc](pmc) |  This folder contains all experiments that are related to articles in Pubmed Central Open Access corpus (PMC). | 
| [environment.yml](environment.yml) | This file contains the necessary packages this repository uses.  |
| [setup.py](setup.py) | This file sets up the annorxiver modules to be used as a regular python package. |


## Set up Environment

Annorxiver uses [conda](http://conda.pydata.org/docs/intro.html) as a python package manager.
Before moving on to the instructions below, please make sure to have it installed.
[Download conda here!!](https://docs.conda.io/en/latest/miniconda.html)
  
Once everything has been installed, type following command in the terminal: 

```bash
bash install.sh
``` 
_Note_: 
There is a bash command within the install.sh that only works on unix systems.
If you are on windows (and possibly macOS), you should remove that file or execute each command individually.

You can activate the environment by using the following command: 

```bash
conda activate annorxiver
```  

## License

This repository is dual licensed as [BSD 3-Clause](LICENSE-BSD.md) and [CC0 1.0](LICENSE-CC0.md), meaning any repository content can be used under either license. This licensing arrangement ensures source code is available under an [OSI-approved License](https://opensource.org/licenses/alphabetical), while non-code content — such as figures, data, and documentation — is maximally reusable under a public domain dedication.
