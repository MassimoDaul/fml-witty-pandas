# Massimo's Embedding Approach

Here are my Jupyter notebooks that show how I explored and created an embedding approach for more accurate retrieval of papers. 
My approach is pretty similar to a baseline of just embedding (title + abstract), but with these differences:

1) I add metadata
2) I use a custom weighted score to combine embeddings from different segments
3) I scan queries and update this weighted score to return more relevant results

Note: When we were using the Arxiv dataset, I scanned the pdf of each paper to extract different sections. But in my exploration, I found that this did not add any more signal than just having the abstract. 
The metadata combined with a good weighted score was more effective.

The layout is:

1) ``ss-build-dataset.ipynb`` this is the script I used to create, match, and index the raw data to be embedded. For consistency and reproducibility
2) ``seg-embeddings-POC.ipynb`` this is my notebook for exploration
3) ``seg-embeddings.ipynb`` this is the final pipeline that embeds and retrives
4) ``results/`` this folder contains the evaluation results from my runs. This can be updated if we want to specificy a different paper set for retrival
