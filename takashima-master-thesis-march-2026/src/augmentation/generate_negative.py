import random
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import pickle

from src.preprocess.extract_edge import create_inference_only_graph

def generate_hard_negatives(original_graph,
                            all_nodes,
                            attack_edges,
                            embedding_matrix,
                            node_to_idx,
                            ratio=0.4):
    """意味的に類似したHard Negativeを生成"""
    hard_negatives = []
    target_count = int(len(attack_edges) * ratio)
    
    # 類似度行列を計算
    similarity_matrix = cosine_similarity(embedding_matrix)
    
    for i, (u, v) in enumerate(attack_edges):
        if u not in node_to_idx or v not in node_to_idx:
            continue
        if i >= target_count:
            break
            
        u_idx = node_to_idx[u]
        v_idx = node_to_idx[v]
        
        # uに意味的に類似したノードを探す
        u_similarities = similarity_matrix[u_idx]
        similar_to_u = np.argsort(u_similarities)[::-1][1:11]  # 上位10個（自分除く）
        
        # vに意味的に類似したノードを探す
        v_similarities = similarity_matrix[v_idx]
        similar_to_v = np.argsort(v_similarities)[::-1][1:11]  # 上位10個（自分除く）
        
        # ランダムに組み合わせを選択
        similar_u_node = all_nodes[random.choice(similar_to_u)]
        similar_v_node = all_nodes[random.choice(similar_to_v)]
        
        # 実際にエッジが存在しないことを確認
        if not original_graph.has_edge(similar_u_node, similar_v_node):
            hard_negatives.append((similar_u_node, similar_v_node))
    
    return hard_negatives

def generate_structural_negatives(original_graph,
                                  attack_edges,
                                  inference_graph,
                                  ratio=0.3):
    """構造的ネガティブサンプリング：グラフ全体での距離2-3のノード対"""
    structural_negatives = []
    target_count = int(len(attack_edges) * ratio)
    
    all_graph_nodes = list(inference_graph.nodes())
    attempts = 0
    max_attempts = target_count * 10  # 効率的な生成のため試行回数制限
    
    while len(structural_negatives) < target_count and attempts < max_attempts:
        u = random.choice(all_graph_nodes)
        v = random.choice(all_graph_nodes)
        
        if u != v and not original_graph.has_edge(u, v) and (u, v) not in structural_negatives:
            try:
                path_length = nx.shortest_path_length(inference_graph, u, v)
                if 2 <= path_length <= 3:
                    structural_negatives.append((u, v))
            except nx.NetworkXNoPath:
                pass
        
        attempts += 1
    
    print(f"構造的ネガティブサンプリング: {len(structural_negatives)} / {target_count} サンプル生成")
    return structural_negatives

def generate_random_negatives(original_graph,
                              attack_edges,
                              all_nodes,
                              ratio=0.3):
    """ランダムネガティブサンプリング"""
    random_negatives = []
    target_count = int(len(attack_edges) * ratio)
    
    while len(random_negatives) < target_count:
        u = random.choice(all_nodes)
        v = random.choice(all_nodes)
        
        if u != v and not original_graph.has_edge(u, v) and (u, v) not in random_negatives:
            random_negatives.append((u, v))
    
    return random_negatives

if __name__ == "__main__":
    # データの読み込み
    filepath = 'data/output/aba_graph_room.pkl'
    with open(filepath, 'rb') as f:
        original_graph = pickle.load(f)
    filepath = 'data/output/inference_graph_room.pkl'
    with open(filepath, 'rb') as f:
        inference_graph = pickle.load(f)
    filepath = 'data/output/attack_edges_room.pkl'
    with open(filepath, 'rb') as f:
        attack_edges = pickle.load(f)
    filepath = 'data/output/node_embeddings_room.pkl'
    with open(filepath, 'rb') as f:
        node_embeddings = pickle.load(f)

    # ノードの取得
    all_nodes = list(inference_graph.nodes())

    # ノードのインデックス
    node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}

    # ハードネガティブの生成
    hard_negatives = generate_hard_negatives(original_graph,
                                            all_nodes,
                                            attack_edges,
                                            node_embeddings,
                                            node_to_idx)

    # 構造的ネガティブの生成
    structural_negatives = generate_structural_negatives(original_graph,
                                                         attack_edges,
                                                         inference_graph)

    # ランダムネガティブの生成
    random_negatives = generate_random_negatives(original_graph,
                                                 all_nodes,
                                                 attack_edges)

    # ネガティブの保存
    with open('data/output/hard_negatives_room.pkl', 'wb') as f:
        pickle.dump(hard_negatives, f)
    with open('data/output/structural_negatives_room.pkl', 'wb') as f:
        pickle.dump(structural_negatives, f)
    with open('data/output/random_negatives_room.pkl', 'wb') as f:
        pickle.dump(random_negatives, f)