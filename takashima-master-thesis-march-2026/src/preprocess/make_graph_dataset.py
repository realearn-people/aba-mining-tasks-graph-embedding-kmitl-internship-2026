import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

def build_aba_graph(aba_frame: pd.DataFrame,
                    contrary_frame: pd.DataFrame,
                    review_title: str | None = None) -> nx.DiGraph:
    """
    概念レベルでのABAグラフを構築する
    
    Args:
        aba_frame: ABAデータフレーム
        contrary_frame: 攻撃関係データフレーム  
        review_title: 特定レビューのみ処理する場合のタイトル
        
    Returns:
        nx.DiGraph: 概念レベルでのABAグラフ
    """
    g = nx.DiGraph()
    
    # レビュー単体のグラフを作成する場合
    if review_title is not None:
        aba_frame = aba_frame[aba_frame["Title"] == review_title]
    
    body_cols = [c for c in aba_frame.columns if c.startswith("Body")]
    cbody_cols = [c for c in aba_frame.columns if c.startswith("Cont.")]

    for _, row in aba_frame.iterrows():
        head = row["Head"]
        
        # 概念レベルでheadノードを追加（レビューIDなし）
        if pd.notna(head):
            g.add_node(head)

        # inference エッジを追加（Body → Head）
        for col in body_cols:
            body_content = row[col]
            if pd.notna(body_content):
                g.add_node(body_content)
                # 同じエッジが複数回追加されても、NetworkXが自動的に重複を処理
                g.add_edge(body_content, head, relation="inference")

        # inference エッジを追加（Contrary Body → Head）  
        for ccol in cbody_cols:
            cont_body_content = row[ccol]
            if pd.notna(cont_body_content):
                g.add_node(cont_body_content)
                g.add_edge(cont_body_content, head, relation="inference")

    # Attack エッジを追加
    for _, r in contrary_frame.iterrows():
        if r["isContrary"]:
            assumption = r["Assumption"]
            proposition = r["Proposition"]
            
            # 概念が実際にグラフに存在する場合のみエッジを追加
            if assumption in g.nodes and proposition in g.nodes:
                g.add_edge(proposition, assumption, relation="attack")
    
    return g

def draw_graph(g: nx.DiGraph, figsize=(10, 6)):
    pos      = nx.spring_layout(g, seed=42)
    supports = [(u, v) for u, v, d in g.edges(data=True) if d["relation"] == "support"]
    attacks  = [(u, v) for u, v, d in g.edges(data=True) if d["relation"] == "attack"]

    plt.figure(figsize=figsize)
    nx.draw_networkx_nodes(g, pos, node_size=700)
    nx.draw_networkx_labels(g, pos, font_size=9)
    nx.draw_networkx_edges(g, pos, edgelist=supports, arrows=True, width=1.5)
    nx.draw_networkx_edges(g, pos, edgelist=attacks,  arrows=True, style="dashed", width=1.5)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    aba_file_path = "data/input/Original ABA Dataset for Version 2 [June 15] - 1. hotel in Larnaca-Cyprus - Topic.csv"
    cols = ['ReviewID', 'Title', 'Topic', 'Pos/Neg', 'Claim', 'Head',
    'Body 1', 'Body 2', 'Body 3', 'Body 4', 'Body 5', 'Body 6', 'Body 7',
    'Body 8', 'Body 9', 'Body 10', 'Body 11', 'Body 12', 'Body 13',
    'Body 14', 'Body 15', 'Cont. Body 1', 'Cont. Body 2', 'Cont. Body 3',
    'Cont. Body 4', 'Cont. Body 5', 'Cont. Body 6', 'Cont. Body 7',
    'Cont. Body 8', 'Cont. Body 9', 'Cont. Body 10', 'Cont. Body 11',
    'Cont. Body 12', 'Cont. Body 13', 'Cont. Body 14', 'Cont. Body 15',]
    aba = pd.read_csv(aba_file_path, usecols=cols)
    aba_room = aba[aba['Topic'] == 'Room']
    # aba_room.to_csv("data/aba_room.csv", index=False)
    
    contra_file_path = "data/output/Silver_Room_ContP_BodyN_4omini.csv"
    contra = pd.read_csv(contra_file_path)
    
    aba_graph_room = build_aba_graph(aba_room, contra)
    
    
