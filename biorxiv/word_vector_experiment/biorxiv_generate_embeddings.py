#!/usr/bin/env python
# coding: utf-8

# # Generate BioRxiv Document Embeddings 

# This notebook is designed to generate document embeddings for every article in bioRxiv.

# In[1]:


from pathlib import Path
import os
import re

from gensim.models import Word2Vec
from gensim.parsing.preprocessing import remove_stopwords
import lxml.etree as ET
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm_notebook
import umap


# In[2]:


journal_map_df = pd.read_csv("../exploratory_data_analysis/output/biorxiv_article_metadata.tsv", sep="\t")
journal_map_df.head()


# # Output Documents to File

# This section dumps all of biorxiv text into a single document in order to train the word2vec model. This is for ease of training the model.

# In[ ]:


parser = ET.XMLParser(encoding='UTF-8', recover=True)

# Only use the most current version of the documents
latest_journal_version = (
    journal_map_df.groupby("doi")
    .agg({"document":"first", "doi":"last"})
)

with open("output/word2vec_input/biorxiv_text.txt", "w") as f:
    for idx, article in tqdm_notebook(latest_journal_version.iterrows()):
        tree = (
            ET.parse(
                open(f"../biorxiv_articles/{os.path.basename(article['document'])}", "rb"),
                parser = parser
            )
        )
        
        root = tree.getroot()
        
        # Grab the abstract text
        abstract_text = root.xpath("//abstract/*//text()")
        abstract_text = list(map(lambda x: remove_stopwords(x), abstract_text))
        f.write("".join(abstract_text))
        
        # Grab the body text
        article_text = root.xpath("//body/sec/*//text()")
        article_text = list(map(lambda x: remove_stopwords(x), article_text))
        f.write("".join(article_text))
        
        #Sign of a new article
        f.write("\n\n")


# # Train Word2Vec

# This section trains the word2vec model (continuous bag of words [CBOW]). Since the number of dimensions can vary I decided to train multiple models: 150, 250, 300. Each model is saved into is own respective directory.

# In[ ]:


class DocIterator: 
    def __init__(self, filepath): 
        self.filepath = filepath 

    def __iter__(self): 
        for line in open(self.filepath, "r"): 
            yield line.split() 


# In[ ]:


word_embedding_sizes = [150, 250, 300]
for size in word_embedding_sizes:
    print(size)
    
    #Create save path
    word_path = Path(f"output/word2vec_models/{size}")
    word_path.mkdir(parents=True, exist_ok=True)
    
    # Run Word2Vec
    words = Word2Vec(DocIterator("output/word2vec_input/biorxiv_text.txt"), size=size, iter=20, seed=100)
    
    #Save the model for future use
    words.save(f"{str(word_path.resolve())}/biorxiv_{size}.model")


# # Generate Document Embeddings

# After training the word2vec models, the next step is to generate a document embeddings. For this experiment each document embedding is generated via an average of all word vectors contained in the document. There are better approaches towards doing this, but this can serve as a simple baseline.

# In[ ]:


def generate_doc_vectors(model, document_df, skip_methods=True):
    document_vec_map = {}
    parser = ET.XMLParser(encoding='UTF-8', recover=True)

    for idx, article in tqdm_notebook(document_df.iterrows()):
        tree = (
                ET.parse(
                    open(f"../biorxiv_articles/{article['document']}", "rb"),
                    parser = parser
                )
            )
        root = tree.getroot()
            
        word_vectors = []
        abstract_text = root.xpath("//abstract/*//text()")
            
        word_vectors += [
            list(
                map(
                    lambda x: model.wv[x], 
                    filter(
                        lambda x: x in model.wv, 
                        text.split(" ")
                    )
                )
            )
            for text in abstract_text
        ]
            
        abstract_vectors = (
            list(
                itertools.chain.from_iterable(
                    filter(lambda x: len(x) > 0, word_vectors)
                )
            )
        )
            
        article_text = root.xpath("//body/sec/*//text()")
            
        word_vectors += [
            list(
                map(
                    lambda x: model.wv[x], 
                    filter(
                        lambda x: x in model.wv, 
                        text.split(" ")
                    )
                )
            )
            for text in article_text
        ]
            
        article_vectors = (
            list(
                itertools.chain.from_iterable(
                    filter(lambda x: len(x) > 0, word_vectors)
                )
            )
        )
            
        total_vectors = abstract_vectors + article_vectors 

        # skips weird documents that don't contain text
        if len(total_vectors) > 0:
            document_vec_map[article['document']] = pd.np.stack(total_vectors).mean(axis=0)

    return document_vec_map


# In[ ]:


for word_model_path in Path().rglob("output/word2vec_models/*/*.model"):
    model_dim = word_model_path.parents[0].stem
    
        
    word_model = Word2Vec.load(str(word_model_path.resolve()))
    biorxiv_vec_map = generate_doc_vectors(word_model, journal_map_df, skip_methods=False)

    biorxiv_vec_df = pd.DataFrame([
        [document] + biorxiv_vec_map[document].tolist()
        for document in biorxiv_vec_map
        ], 
        columns=["document"] + list(map(lambda x: f"feat_{x}", range(int(model_dim))))
    )
    
    biorxiv_vec_df.to_csv(
        f"output/word2vec_output/biorxiv_all_articles_{model_dim}.tsv.xz", 
        sep="\t", index=False,
        compression="xz"
    )


# # UMAP the Documents

# After generating document embeddings, the next step is to visualize all the documents. In order to visualize the embeddings a low dimensional representation is needed. UMAP is an algorithm that can generate this representation, while grouping similar embeddings together. 

# In[ ]:


random_state = 100
n_neighbors = journal_map_df.category.unique().shape[0]
n_components = 2


# In[ ]:


for biorxiv_doc_vectors in Path().rglob("output/word2vec_output/biorxiv_all_articles*.tsv.xz"):
    model_dim = int(re.search(r"(\d+)", biorxiv_doc_vectors.stem).group(1))
    biorxiv_articles_df = pd.read_csv(str(biorxiv_doc_vectors.resolve()), sep="\t")
    
    reducer = umap.UMAP(
        n_components=n_components, 
        n_neighbors=n_neighbors, 
        random_state=random_state
    )
    
    embedding = reducer.fit_transform(
        biorxiv_articles_df[[f"feat_{idx}" for idx in range(model_dim)]].values
    )
    
    umapped_df = (
        pd.DataFrame(embedding, columns=["umap1", "umap2"])
        .assign(document=biorxiv_articles_df.document.values.tolist())
        .merge(journal_map_df[["category", "document", "doi"]], on="document")
    )
    
    umapped_df.to_csv(
        f"output/embedding_output/umap/biorxiv_umap_{model_dim}.tsv", 
        sep="\t", index=False
    )


# # TSNE the Documents

# After generating document embeddings, the next step is to visualize all the documents. In order to visualize the embeddings a low dimensional representation is needed. TSNE is an another algorithm (besides UMAP) that can generate this representation, while grouping similar embeddings together.  

# In[ ]:


n_components = 2
random_state = 100


# In[ ]:


for biorxiv_doc_vectors in Path().rglob("output/word2vec_output/biorxiv_all_articles*.tsv.xz"):
    model_dim = int(re.search(r"(\d+)", biorxiv_doc_vectors.stem).group(1))
    biorxiv_articles_df = pd.read_csv(str(biorxiv_doc_vectors.resolve()), sep="\t")
    
    reducer = TSNE(n_components=n_components, random_state=random_state)
    
    embedding = reducer.fit_transform(
        biorxiv_articles_df[[f"feat_{idx}" for idx in range(model_dim)]].values
    )
    
    tsne_df = (
        pd.DataFrame(embedding, columns=["tsne1", "tsne2"])
        .assign(document=biorxiv_articles_df.document.values.tolist())
        .merge(journal_map_df[["category", "document", "doi"]], on="document")
    )

    tsne_df.to_csv(
        f"output/embedding_output/tsne/biorxiv_tsne_{model_dim}.tsv", 
        sep="\t", index=False,
    )


# # PCA the Documents

# In[3]:


n_components = 2
random_state = 100


# In[4]:


for biorxiv_doc_vectors in Path().rglob("output/word2vec_output/biorxiv_all_articles*.tsv.xz"):
    model_dim = int(re.search(r"(\d+)", biorxiv_doc_vectors.stem).group(1))
    biorxiv_articles_df = pd.read_csv(str(biorxiv_doc_vectors.resolve()), sep="\t")
    
    reducer = PCA(
        n_components = n_components,
        random_state = random_state
    )
    
    embedding = reducer.fit_transform(
        biorxiv_articles_df[[f"feat_{idx}" for idx in range(model_dim)]].values
    )
    
    pca_df = (
        pd.DataFrame(embedding, columns=["pca1", "pca2"])
        .assign(document=biorxiv_articles_df.document.values.tolist())
        .merge(journal_map_df[["category", "document", "doi"]], on="document")
    )
    
    pca_df.to_csv(
        f"output/embedding_output/pca/biorxiv_pca_{model_dim}.tsv",
        sep="\t", index=False
    )

