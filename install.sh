conda env create --file environment.yml
source ~/anaconda3/etc/profile.d/conda.sh
conda activate annorxiver
python -m spacy download en_core_web_sm
pip install -e .
