import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

configs = [
    (100, 0.1),   
    (500, 0.02),  
    (1000, 0.01)  
]

num_runs = 30

results = []

for config_idx, (n, p) in enumerate(configs, 1):
    print("Configuration " + str(config_idx) + ": n = " + str(n) + ", p = " + str(p))
    
    avg_degrees = []
    avg_clustering_coeffs = []
    avg_path_lengths = []
    degree_distributions = []
    
    for run in range(num_runs):
        G = nx.erdos_renyi_graph(n, p)
        
        degrees = [d for _, d in G.degree()]
        avg_degree = np.mean(degrees)
        avg_degrees.append(avg_degree)
        
        avg_clustering = nx.average_clustering(G)
        avg_clustering_coeffs.append(avg_clustering)
        
        if nx.is_connected(G):
            avg_path_length = nx.average_shortest_path_length(G)
        else:
            largest_cc = max(nx.connected_components(G), key=len)
            G_largest = G.subgraph(largest_cc).copy()
            avg_path_length = nx.average_shortest_path_length(G_largest)
        avg_path_lengths.append(avg_path_length)
        
        degree_distributions.extend(degrees)
    
    mean_avg_degree = np.mean(avg_degrees)
    mean_avg_clustering = np.mean(avg_clustering_coeffs)
    mean_avg_path_length = np.mean(avg_path_lengths)
    
    results.append({
        'n': n,
        'p': p,
        'mean_avg_degree': mean_avg_degree,
        'mean_avg_clustering': mean_avg_clustering,
        'mean_avg_path_length': mean_avg_path_length,
        'degree_distribution': degree_distributions
    })
    
    print("Average Degree (over " + str(num_runs) + " runs): " + str(round(mean_avg_degree, 4)))
    print("Average Clustering Coefficient (over " + str(num_runs) + " runs): " + str(round(mean_avg_clustering, 4)))
    print("Average Path Length (over " + str(num_runs) + " runs): " + str(round(mean_avg_path_length, 4)))

plt.figure(figsize=(10, 6))
for config_idx, result in enumerate(results, 1):
    degrees = result['degree_distribution']
    mean_degree = result['mean_avg_degree']
    plt.hist(degrees, bins=30, density=True, alpha=0.5, label="Config " + str(config_idx) + " (n=" + str(result['n']) + ", p=" + str(result['p']) + ")")
    plt.axvline(mean_degree, linestyle='--', color='black', alpha=0.5)

plt.title("Degree Distribution for Different Configurations")
plt.xlabel("Degree")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.show()