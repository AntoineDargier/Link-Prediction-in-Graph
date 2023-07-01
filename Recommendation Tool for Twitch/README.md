# Recommendation Tool for Twitch

### Goal
Create a recommendation tool for Twitch using link prediction in graph

### Language
````Python```

### Contents
1. Problem definition
2. Dataset
3. Graph similarity method
4. Deepwalk method
5. GNN method
6. Evaluation

### Librairies
* ```networkx```
* ```numpy```
* ```matplotlib```
* ```random```
* ```scikit-learn```
* ```pandas```
* ```gensim```
* ```seaborn```
* ```Pytorch```

  ### Conclusion
I implemented three different algorithms dedicated to link prediction: Graph similarity with different kinds of features (older ones and more recent ones), deepwalk/skipgram methods and Graph Neural Networks (GNN).
These methods have been applied in a real-life case of link prediction on the social media Twitch based on a publically available dataset depicting mutual "follow" relationships between a subset of Twitch users during the year 2018.
I found very satisfactory results for Graph Similarity methods, in particular while using classical features (Degree centrality for both extremity nodes, Betweenness centrality, Jaccard coefficient, Preferential attachment and Adamic Adar index) or a more recent one called Direct-Indirect Common Neighbours, which succeed in maximizing both the overall accuracy (80%) and the number of True Positive (580). Similar results were also obtained with Deepwalk method, with an overall accuracy just behind it.
