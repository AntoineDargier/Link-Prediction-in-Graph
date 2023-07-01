import csv
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


###################
#  Used Functions #
###################

# The following list contains the index of the more interesting node features
# (ie. node features where at least 3% of the data do have the futures and at least 3% does not)
features_index = [21, 49, 61, 77, 80, 92, 111, 127, 129, 132, 138, 144, 252, 263, 268, 334, 356, 369, 385, 387, 481, 499, 521, 570, 677, 704]



def data_import():
    """
    This function allows us to gather all the data from the Kaggle dataset in two forms : a networkx graph and lists

    Returns:
        graph: A netxorkx graph containing all nodes and their annotations as well as edges between them, if they exist
        train_test: A list of lists. Each of these lists contain the id of the two nodes of an edge, and a 1 if the edge exists or 0 if it does not
        test_set: A list of lists. Each of these lists contain the id of the two nodes to consider for link prediction
        annotations: A list of lists. Each of these lists contain the 952 features of each node
    """

    # Reading of the train.txt file
    with open("train.txt", "r") as f:
        reader = csv.reader(f)
        train_set = list(reader)

    # Creation of the train_set output
    train_set = [element[0].split(" ") for element in train_set]

    # Reading of the test.txt file
    with open("test.txt", "r") as f:
        reader = csv.reader(f)
        test_set = list(reader)

    # Creation of the test_set output
    test_set = [element[0].split(" ") for element in test_set]

    # Creation of the output graph
    graph = nx.Graph()
    annotations = []

    for edge in train_set:
        node_1 = int(edge[0])
        node_2 = int(edge[1])

        # Add the nodes to graph if they do not exist
        graph.add_node(node_1)
        graph.add_node(node_2)

        value = int(edge[2])

        # Creation of the annotations output
        annotations.append(value)

        # If necessary, adds an edge between the nodes in the graph
        if bool(value):
            graph.add_edge(node_1, node_2)

    # Reading of the node_information.csv file
    with open("node_information.csv", "r") as f:
        reader = csv.reader(f)
        info_set = list(reader)

    features_dic = {}

    # Reshaping of the data by creating a list of features
    for info in info_set:

        node = int(info[0])

        features = info[1:]
        N_features = len(features)

        features_node = {}

        for i in range(N_features):

            features_node[i + 1] = int(float(features[i]))

        features_dic[node] = features_node

    # Adding the node features to the networkx graph
    nx.set_node_attributes(graph, features_dic)

    return graph, train_set, test_set, annotations


def feature_extractor(data, graph):
    """
    Creates a feature vector for each edge of the graph contained in samples

    Returns:
        feature_vector : a list of arrays. Each array contains the different features calculated for each edge
    """

    # Creation of the feature
    feature_vector = []

    # Degree Centrality measure
    deg_centrality = nx.degree_centrality(graph)

    # Betweeness centrality measure
    betweeness_centrality = nx.betweenness_centrality(graph)

    for edge in data:

        # Reading the source and target nodes of each edge
        source_node, target_node = int(edge[0]), int(edge[1])

        # Reading the features of each of this node
        source_node_features, target_node_features = list(
            graph.nodes[source_node].values()
        ), list(graph.nodes[target_node].values())

        # Keeping only the more interesting features
        source_temp = [source_node_features[x] for x in features_index]
        target_temp = [target_node_features[x] for x in features_index]

        source_node_features = source_temp
        target_node_features = target_temp

        # Degree Centrality
        source_degree_centrality = deg_centrality[source_node]
        target_degree_centrality = deg_centrality[target_node]

        # Betweeness centrality measure
        diff_bt = (
            betweeness_centrality[target_node] - betweeness_centrality[source_node]
        )

        # Preferential Attachement
        pref_attach = list(
            nx.preferential_attachment(graph, [(source_node, target_node)])
        )[0][2]

        # AdamicAdar
        if source_node == target_node:
            aai = 0
        else:
            aai = list(nx.adamic_adar_index(graph, [(source_node, target_node)]))[0][2]

        # Jaccard
        jacard_coeff = list(
            nx.jaccard_coefficient(graph, [(source_node, target_node)])
        )[0][2]

        # Create edge feature vector with all metric computed above. First line : with node features / Second line : without node features

        # feature_vector.append(np.array(source_node_features + target_node_features + [source_degree_centrality, target_degree_centrality,
        #                                diff_bt, pref_attach, aai, jacard_coeff]) )
        feature_vector.append(
            np.array(
                [
                    source_degree_centrality,
                    target_degree_centrality,
                    diff_bt,
                    pref_attach,
                    aai,
                    jacard_coeff,
                ]
            )
        )

    return feature_vector


def update_with_kneighbors(nn, i, set_, test_features):
    """
    For all the wanted edges, recalculate the predicted class with k-Neighbours and updates the output array with the majoritarian class of the k closest neighbours of the edge.

    Returns:
        a 1 or 0 according to the output of the k-Neighbours algorithm
    """

    # Inport the features of the wanted edge
    edge_features = [test_features[i]]

    # Creation of the list of predictions with k-Neighbours algorithm
    nn_predict = nn.kneighbors(edge_features)
    nn_predict = nn_predict[1][0]

    annotation_list = []

    for index in nn_predict:
        annotation_list.append(set_[index])

    # Checking which class has the majority of appearances in the k Nearest Neighbours
    return int((sum(annotation_list) / len(annotation_list)) >= 0.5)

def prediction(
    graph, train_set, test_set, train_labels, n_neighbors=100, threshold=0.58
):
    """
    Function used to compute the final predictions on the testing dataset

    Returns :
        train_preds : a list of 1 or 0 according to the final prediction of our model for each edge of the testing dataset
    """

    # Import of the features of each edge of the training dataset
    train_features = feature_extractor(train_set, graph)

    # IF THE NODE FEATURES ARE INCLUDED, import the features of each edge and swap the features from the source and the target node, to take into account the fact that edges are non-directed
    # train_features_reversed = [np.concatenate([arr[len(features_index):2*len(features_index)], arr[:len(features_index)], arr[2*len(features_index):]]) for arr in train_features]

    # Import of the features of each edge of the testing dataset
    test_features = feature_extractor(test_set, graph)

    # Choose one model among the following ones :
    clf = LogisticRegression(max_iter=1000, n_jobs=8, tol=1e-7)
    # clf = RandomForestClassifier(n_estimators = 400, n_jobs = 8)
    # clf = SGDClassifier(loss="modified_huber", max_iter = 5000, n_jobs = 8, alpha = 0.01)
    # clf = GradientBoostingClassifier()

    # Training the model with features and testing annotations
    clf.fit(train_features, train_labels)

    # IF NECESSARY, training the model on the reversed dataset
    # clf.fit(train_features_reversed, train_labels)

    # Creating a Nearest-Neighbors instance and feeding it with all features
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(train_features)
    # nn.fit(train_features_reversed)

    # Applying the model to the testing data
    prob_preds = clf.predict_proba(test_features)
    train_preds = clf.predict(test_features)

    counter = 0

    # For each tested edge, check if the prediction probability is above the threshold. If not, stores the index of the edge and send it to the k-Neighbors algorithm for re-calculation.
    for i, pred in enumerate(train_preds):
        pred = train_preds[i]

        if prob_preds[i][pred] < threshold:

            counter += 1
            train_preds[i] = update_with_kneighbors(nn, i, train_labels, test_features)

    print(
        100 * counter / len(train_preds)
    )  # Percentage of edges predicted with enough confidence in the first round
    return train_preds


def parameters_finetuning(
    graph, train_set, validation_set, annotations_train, annotations_validation
):
    """
    Function used to finetune some parameters. Here configured to finetune the value of the threshold and the number of neighbours considered for the k-Neighbours algorithm.

    Returns:
        errors: A list containing the error calculated for each combination of parameters trained
    """

    annotations_validation = np.array(annotations_validation)

    # Insert here the range of parameters that should be tested
    N_neighbors = [800, 850, 900, 950, 975, 1000, 1025, 1050, 1100, 1150, 1200]
    thresholds = [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]

    # Test all the possible combinations of the pre-determined parameters
    for threshold in thresholds:

        errors = []

        for n_neighbors in N_neighbors:

            predictions = prediction(
                graph,
                train_set,
                validation_set,
                annotations_train,
                n_neighbors=n_neighbors,
                threshold=threshold,
            )
            errors.append(np.linalg.norm(predictions - annotations_validation, ord=1))

        plt.plot(N_neighbors, errors)

    plt.legend(thresholds)

    # Plots the error according all tested parameters
    plt.show()

    return errors


if __name__ == "__main__":
    # Creation of our dataset
    graph, train_set, test_set, annotations = data_import()

    # IF WANTED, finetuning of hyperparameters
    # train_set, validation_set, annotations_train, annotations_validation = train_test_split(train_set, annotations, test_size = 0.1, random_state = 42)
    # parameters_finetuning(graph, train_set, validation_set, annotations_train, annotations_validation)

    # Computing predictions according to our model
    prediction_list = prediction(graph, train_set, test_set, annotations, 1000, 0.62)

    # Reshaping our list to the desired format
    predictions = np.array(prediction_list)
    predictions = zip(np.array(range(len(test_set))), predictions)

    # Writing it to the output file
    with open("predictions.csv", "w") as pred:
        csv_out = csv.writer(pred)
        csv_out.writerow(i for i in ["ID", "Predicted"])
        for row in predictions:
            csv_out.writerow(row)
        pred.close()
