import pandas as pd
import networkx as nx
from networkx.algorithms import community
import matplotlib.pyplot as plt
from itertools import combinations


df = pd.read_csv('/Users/ayushjain/Downloads/inst414 A2 reddit WSB data.csv')

# Filter out posts (rows where 'title' is not null)
comments_df = df[df['title'].isna()]

# Exclude bot/mod accounts
exclude_users = ['VisualMod', 'AutoModerator', '[deleted]']
comments_df = comments_df[~comments_df['author'].isin(exclude_users)]

# Seting limits due to speed constraints (first 500 posts, first 20 commenters per post)
MAX_POSTS = 500
MAX_AUTHORS_PER_POST = 20 

post_ids = comments_df['post_id'].unique()[:MAX_POSTS]

# Building the graph
G = nx.Graph()

for post_id in post_ids:
    authors = comments_df[comments_df['post_id'] == post_id]['author'].unique()
    authors = [a for a in authors if a not in exclude_users][:MAX_AUTHORS_PER_POST]
    
    for a1, a2 in combinations(authors, 2):
        if G.has_edge(a1, a2):
            G[a1][a2]['weight'] += 1
        else:
            G.add_edge(a1, a2, weight=1)

print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")

# Finding top 3 influential users
centrality = nx.degree_centrality(G)
top_users = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:3]

print("\nTop 3 influential users by Degree Centrality:")
for user, score in top_users:
    print(f"{user}: {score:.4f}")

# Scale degree centrality for better visualization in Gephi
scaled_centrality = {node: val*100 for node, val in centrality.items()}
nx.set_node_attributes(G, scaled_centrality, "degree_centrality_scaled")  # new attribute


# Add top 3 attribute for labeling in Gephi
top3_nodes = [user for user, _ in top_users]
for node in G.nodes():
    if node in top3_nodes:
        G.nodes[node]['top3_label'] = node
    else:
        G.nodes[node]['top3_label'] = ""

# PageRank
pagerank = nx.pagerank(G)
top_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:3]

print("\nTop 3 users by PageRank:")
for user, score in top_pagerank:
    print(f"{user}: {score:.4f}")

# Betweenness Centrality
betweenness = nx.betweenness_centrality(G)
top_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:3]

print("\nTop 3 users by Betweenness Centrality:")
for user, score in top_betweenness:
    print(f"{user}: {score:.4f}")

# Detecting communities (clusters)
communities = community.greedy_modularity_communities(G)
print(f"\nDetected {len(communities)} communities")
for i, comm in enumerate(list(communities)[:3]):
    print(f"Community {i+1} sample: {list(comm)[:10]}")


# Exporting graph for Gephi
# Add degree centrality as a node attribute
nx.set_node_attributes(G, centrality, "degree_centrality")

# Add community labels
for i, comm in enumerate(communities):
    for node in comm:
        G.nodes[node]["community"] = i

# Export as GEXF (Gephi compatible)
nx.write_gexf(G, "/Users/ayushjain/Downloads/inst414_module2_reddit_finance_top3.final.gexf")
print("\nGraph exported")
