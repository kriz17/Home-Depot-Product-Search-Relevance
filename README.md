# Home-Depot-Product-Search-Relevance
Predicting the relevance of search results on homedepot.com

Please find a detailed blog on this case study here: 

Project Description:
In this project, I propose a robust way of predicting the desired products for a given search query, using techniques involving machine learning, natural language processing, and information retrieval.
I explored a variety of techniques from the field of Information Retrieval to extract features from the data provided. The techniques ranged from simple set operators to Latent Semantic Indexing. LSI based similarities and traditional retrieval models such as BM25 and Language Models have proven to be extremely effective. For modeling, I built a Stacking Regressor with 17 base models and Ridge Regressor as the meta model which got a final score of 0.4652 on Kaggle's test set.

Problem Statement
The task is simple to understand. For any search query that the customer enters, I need to find the most relevant products and show them to the user in order of their relevance. From a business point of view, there are a few points which need to be considered. First, the products need to be ranked, thus, even among the most relevant products, we need to be able to tell which one is more relevant. Second, there is a time constraint involved i.e. the results need to be shown within seconds.
Machine Learning Formulation of the Business Problem
The task can be formulated as follows: Given a search and a product, find the relevance score between them i.e. how relevant that product is to the search query at hand. So let's say my machine has learned how to predict the relevance score for a (search-query, product) pair. Now, for any search that the user enters, I can calculate the relevance score for that very search paired with all the products in my database and show the (say) top 10 results to the customer.
Thus, if I have a lot of labeled data i.e. a lot of (search-query, product) pairs with their relevance scores then I can pose this as a supervised ML problem. And that is exactly what I have done in this case study. The data I have used is provided by Home Depot for the Kaggle competition home-depot-product-search-relevance
Now in a real-world e-commerce search engine, calculating the relevance score for every product for a given search is not possible, because, in any typical e-commerce website, the number of products is very large, and hence it's computationally expensive and very time-consuming.
✦ Thus, first, we retrieve a few candidate products using a simpler retrieval model which permits fast query evaluation. And in the second phase, a more accurate but computationally expensive machine-learned model is used to re-rank these products.
To explain this, consider we have a set of 1,00,000 products. And the search query is "solar lamp". The first simpler retrieval model will retrieve a few candidate products. This model can be as simple as an AND operator between the words of search and product-text. So here a model governed by AND operator will get all the products with the words "solar" and "lamp". Say it retrieves some 500 products. Now on top of these 500 products, we can run our complex machine learning algorithm and calculate the relevance score for each (search, product) pair where our search term "solar lamp" remains constant and products vary. And now we re-rank them based on their relevance score and show the top products to the user. This is called Learning To Rank (LTOR). In this blog, the focus will be on the learning-to-rank part of the system. In the end, I have also explained how I extended it to make a full-fledged search engine.

![image](https://user-images.githubusercontent.com/46672597/123036603-f7f93d00-d40a-11eb-968f-b60b546b1501.png)

