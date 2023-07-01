# Link Prediction Kaggle Challenge

### Goal
Link prediction problem on a leaky graph given as input data

### Language
```Python```

### Contents
1. Graph exploration
2. Feature engineering
3. Our solution: models and parameters tuning
4. Results

### Librairies
* ```networkx```
* ```scikit-learn```
* ```numpy```
* ```matplotlib```

### Conclusion
My best solution consisting of a Logistic Regression and a k-Neighbours combination allowed me to reach my final best score of 76.29%.

This project shows us particular problems that we encounter
when we deal with graphs which are very particular. We need to
understand well the structure of the graph and the nature of the
links to model our graph well. We see that we very specific graphs,
it can be very difficult to have very good results because some links
can be unpredictable, or information can be lost if e.g part of the
graph is no more part of the same connected component.
