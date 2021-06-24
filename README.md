# Home-Depot-Product-Search-Relevance
Predicting the relevance of search results on homedepot.com

Please find a detailed blog on this case study here: 

### Project Summary:
In this project, I propose a robust way of predicting the desired products for a given search query, using techniques involving machine learning, natural language processing, and information retrieval. The data used is from the Kaggle Competetion [home-depot-product-search-relevance](https://www.kaggle.com/c/home-depot-product-search-relevance). The most importatn part of the project was feature engineering, for which a variety of techniques from the field of Information Retrieval were. The techniques ranged from simple set operators to Latent Semantic Indexing. LSI based similarities and traditional retrieval models such as BM25 and Language Models have proven to be extremely effective. For modeling, I built a Stacking Regressor with 17 base models and Ridge Regressor as the meta model which got a final score of 0.4652 on Kaggle, ranking it in the top 10%. Further, I extended this to a full-fledged search engine by adding an initial retrieval model in the form of BM25.

### Full Pipeline
![image](https://user-images.githubusercontent.com/46672597/123036603-f7f93d00-d40a-11eb-968f-b60b546b1501.png)

### Web Application Demo
[![Web Application Demo](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://www.youtube.com/watch?v=kuV3HeUN8-Y)


