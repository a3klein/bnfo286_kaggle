import torch
import pickle
import argparse
import numpy as np
import pandas as pd

import igraph as ig
import leidenalg as la

import torch.nn.functional as F
from torch_geometric.nn.pool import knn
from torch_geometric.nn import Node2Vec
from scipy.sparse import coo_matrix

# Learn the node2vec embeddings the the proteins in the PPI
class Node2VecEmbeddings(object):
    def __init__(self, network_edgelist, confidence, embedding_dim, walk_length, context_size, walks_per_node, epochs, lr, batch_size):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load in network and filter edges based on confidence (if any)
        network = pd.read_csv(network_edgelist, names=['source', 'target', 'weight'], sep = '\t')
        network = network[(network["weight"] > confidence)]
        
        # Create int IDs for the protein label, create edge_index and edge_weigth lists (needed for node2vec)
        unique_labels = pd.unique(network[['source','target']].values.ravel())
        label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        network['source'] = network['source'].map(label_to_id)
        network['target'] = network['target'].map(label_to_id)
        edge_index = torch.tensor(network[['source', 'target']].values, dtype=torch.long).t()

        # Instantiate the node2vec model
        self.model = Node2Vec(edge_index=edge_index, 
                              embedding_dim=embedding_dim,
                              walk_length=walk_length, # walks to sample for each node
                              context_size=context_size, # skip-gram window size, increases sampling rate
                              walks_per_node=walks_per_node, # walks to sample for each node
                              p=1.0,
                              q=1.0,
                              num_negative_samples=1,
                              sparse=True).to(self.device)
        
        self.optimizer = torch.optim.SparseAdam(list(self.model.parameters()), lr=lr)
        self.id_to_label = {idx: label for label, idx in label_to_id.items()}
        self.num_nodes = len(unique_labels)
        self.batch_size = batch_size
        self.epochs = epochs

    # Train node2vec over range of epochs to acquire node embeddings
    def train_node2vec(self):

        # Train node2vec model
        for epoch in range(self.epochs):
            loss = self.train()
            print(f"Epoch: {epoch}, Loss: {loss:.4f}")

        # Detach embeddings for GPU
        self.model.eval()
        embeddings = self.model().detach().cpu().numpy()

        # Convert embeddings to a DataFrame indexed by the node name
        embeddings = pd.DataFrame(embeddings, index=[self.id_to_label[i] for i in range(self.num_nodes)])

        return(embeddings)

    # Training function
    def train(self):
        
        self.model.train()
        total_loss = 0
        loader = self.model.loader(batch_size=self.batch_size, shuffle=True)

        # The model provides a loader that returns pairs (pos_rw, neg_rw)
        #   pos_rw: random walks for positive edges
        #   neg_rw: random walks for negative edges
        for pos_rw, neg_rw in loader:
            pos_rw, neg_rw = pos_rw.to(self.device), neg_rw.to(self.device)
            self.optimizer.zero_grad()
            loss = self.model.loss(pos_rw, neg_rw)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
    
        return(total_loss / len(loader))


# Cluster the embeddings learned from node2vec. For now, we will cluster with Leiden, but the embeddings could
#   be clustered with any method, hence we can expand additional clustering methods into the class.
class ClusterEmbeddings(object):
    def __init__(self, embeddings, seed=42):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.node_names = embeddings.index.tolist()
        self.embeddings = torch.from_numpy(embeddings.values).to(device=self.device)

    def Leiden(self, num_neighbors=10, resolution=1.0, cosine_sim=False, seed=42):

        N = len(self.embeddings)

        # Create KNN graph based on cosine distance, otherwise use Euclidean distance
        if cosine_sim:
            row, col = knn(x=self.embeddings, y=self.embeddings, k=num_neighbors, cosine=cosine_sim)
            cos_sim = F.cosine_similarity(self.embeddings[row], self.embeddings[col], dim=1)
            cos_sim = torch.clamp(cos_sim, min=0.0, max=1.0)
            row_cpu = row.detach().cpu().numpy()
            col_cpu = col.detach().cpu().numpy()
            sim_cpu = cos_sim.detach().cpu().numpy()
            knn_dist_mat = coo_matrix((sim_cpu, (row_cpu, col_cpu)), shape=(N, N))
        else:
            row, col = knn(x=self.embeddings, y=self.embeddings, k=num_neighbors, cosine=cosine_sim)
            distances = (self.embeddings[row] - self.embeddings[col]).norm(dim=1, p=2)
            row_cpu = row.detach().cpu().numpy()
            col_cpu = col.detach().cpu().numpy()
            dist_cpu = distances.detach().cpu().numpy()
            knn_dist_mat = coo_matrix((dist_cpu, (row_cpu, col_cpu)), shape=(N, N))
                 
        # Convert to igraph object and run the Leiden algorithm
        graph = ig.Graph.Weighted_Adjacency(knn_dist_mat, mode="directed", loops=False)
        partition = la.find_partition(graph, la.RBConfigurationVertexPartition, weights="weight", 
                                      resolution_parameter=resolution, seed=seed)

        cluster_labels = []
        for cluster in partition.membership:
            cluster_labels.append("cluster"+(str(cluster+1)))

        # Write the cluster results to Kaggle submission format
        cluster_results = pd.DataFrame({"id": list(range(len(self.node_names))), 
                                        "xxx": self.node_names, 
                                        "prediction": cluster_labels})

        return(cluster_results)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Network edgelist from Kaggle and confidence filter
    parser.add_argument('--network_edgelist', dest='network_edgelist', type=str, default='data/interaction/network_with_weights.edgelist')
    parser.add_argument('--confidence', dest='confidence', type=float, default=0.5)

    # Node2Vec Hyperparameters
    parser.add_argument('--embedding_dim', dest='embedding_dim', type=int, default=128)
    parser.add_argument('--walk_length', dest='walk_length', type=int, default=80)
    parser.add_argument('--context_size', dest='context_size', type=int, default=10)
    parser.add_argument('--walks_per_node', dest='walks_per_node', type=int, default=5)
    parser.add_argument('--epochs', dest='epochs', type=int, default=50)
    parser.add_argument('--lr', dest='lr', type=float, default=0.01)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=128)

    # Leiden Clustering Hyperparameters
    parser.add_argument('--num_neighbors', dest='num_neighbors', type=int, default=10)
    parser.add_argument('--resolution', dest='resolution', type=float, default=1.0)
    parser.add_argument('--cosine_sim', dest='cosine_sim', action='store_true', default=False)

    # File for saving node2vec embeddings and cluster results 
    parser.add_argument('--embedding_out', dest='embedding_out', type=str, default="node2vec_embeddings.pkl")
    parser.add_argument('--cluster_out', dest='cluster_out', type=str, default="protein_clusters.csv")
    args = parser.parse_args()

    # Compute node2vec embeddings
    print("Learning Node2Vec Embeddings ...")
    embeddings = Node2VecEmbeddings(network_edgelist=args.network_edgelist,
                                    confidence=args.confidence,
                                    embedding_dim=args.embedding_dim,
                                    walk_length=args.walk_length,
                                    context_size=args.context_size,
                                    walks_per_node=args.walks_per_node,
                                    epochs=args.epochs,
                                    lr=args.lr,
                                    batch_size=args.batch_size).train_node2vec()

    print('\n')

    # Save the node2vec embeddings to pickle
    with open(args.embedding_out, "wb") as handle:
        pickle.dump(embeddings, handle)

    # Compute clusters from the nod2vec embeddings (here with Leiden)
    print("Clustering the Node2Vec Embeddings ...")
    ce = ClusterEmbeddings(embeddings=embeddings)
    clusters = ce.Leiden(num_neighbors=args.num_neighbors, resolution=args.resolution, cosine_sim=args.cosine_sim)

    # Save cluster results to csv
    clusters.to_csv(args.cluster_out, index = False)

    
        