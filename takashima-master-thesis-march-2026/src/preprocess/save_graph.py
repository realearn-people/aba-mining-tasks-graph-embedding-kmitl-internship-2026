import pickle
import os
import pandas as pd
import networkx as nx
from make_graph_dataset import build_aba_graph

def save_graph_pickle(graph: nx.DiGraph, filename: str):
    """グラフをPickle形式で保存"""
    # プロジェクトルートを推定し、出力パスを絶対指定
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_dir = os.path.join(project_root, "data", "output")
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{filename}.pkl")
    
    with open(filepath, 'wb') as f:
        pickle.dump(graph, f)
    
    print(f"✓ グラフを保存しました: {filepath}")
    print(f"  - ノード数: {graph.number_of_nodes()}")
    print(f"  - エッジ数: {graph.number_of_edges()}")
    
    # ファイルサイズも表示
    file_size = os.path.getsize(filepath)
    print(f"  - ファイルサイズ: {file_size:,} bytes")
    
    return filepath

if __name__ == "__main__":
    # プロジェクトルートを推定
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # データ読み込み
    #aba_file_path = os.path.join(project_root, "data/input/Original ABA Dataset for Version 2 [June 15] - 1. hotel in Larnaca-Cyprus - Topic.csv")
    aba_file_path = os.path.join(project_root, "data/input/aba_nodes.csv")
    # Room/Staff の4CSVを対象（存在するもののみ使用）
    '''room_contp_bodyn = os.path.join(project_root, "data/output/Silver_Room_ContP_BodyN_4omini.csv")
    room_contn_bodyp = os.path.join(project_root, "data/output/Silver_Room_ContN_BodyP_4omini.csv")
    staff_contp_bodyn = os.path.join(project_root, "data/output/Silver_Staff_ContP_BodyN_4omini.csv")
    staff_contn_bodyp = os.path.join(project_root, "data/output/Silver_Staff_ContN_BodyP_4omini.csv")'''
    silver_files = [
    # PN sheet: Contrary(P)Body(N) — positive contrary assumption, negative body
    "data/output/Silver_Staff_ContP_BodyN_4omini.csv",
    "data/output/Silver_Price_ContP_BodyN_4omini.csv",
    "data/output/Silver_CheckIn_ContP_BodyN_4omini.csv",
    "data/output/Silver_CheckOut_ContP_BodyN_4omini.csv",
    # NP sheet: Contrary(N)Body(P) — negative contrary assumption, positive body
    "data/output/Silver_Staff_ContN_BodyP_4omini.csv",
    "data/output/Silver_Price_ContN_BodyP_4omini.csv",
    "data/output/Silver_CheckIn_ContN_BodyP_4omini.csv",
    "data/output/Silver_CheckOut_ContN_BodyP_4omini.csv",
    # PP sheet: Contrary(P)Body(P) — both positive, always NOT_CONTRARY
    "data/output/Silver_Staff_ContP_BodyP_4omini.csv",
    "data/output/Silver_Price_ContP_BodyP_4omini.csv",
    "data/output/Silver_CheckIn_ContP_BodyP_4omini.csv",
    "data/output/Silver_CheckOut_ContP_BodyP_4omini.csv",
    # NN sheet: Contrary(N)Body(N) — both negative, always NOT_CONTRARY
    "data/output/Silver_Staff_ContN_BodyN_4omini.csv",
    "data/output/Silver_Price_ContN_BodyN_4omini.csv",
    "data/output/Silver_CheckIn_ContN_BodyN_4omini.csv",
    "data/output/Silver_CheckOut_ContN_BodyN_4omini.csv",
]
    
    #contra_paths = [room_contp_bodyn, room_contn_bodyp, staff_contp_bodyn, staff_contn_bodyp]
    contra_paths = silver_files
    contra_list = []
    for p in contra_paths:
        if os.path.exists(p):
            print(f"📂 Contraファイルを読み込み: {p}")
            df = pd.read_csv(p)
            print(f"  - データ数: {len(df)} 行")
            contra_list.append(df)
        else:
            print(f"⚠️ Contraファイルが見つかりませんでした: {p}")
    
    if not contra_list:
        raise FileNotFoundError("対象のContra CSVが1つも読み込めませんでした。パスを確認してください。")

    # 複数CSVを結合
    contra_combined = pd.concat(contra_list, ignore_index=True) if len(contra_list) > 1 else contra_list[0]
    
    # 重複チェック（もしあれば）
    duplicates = contra_combined.duplicated().sum()
    if duplicates > 0:
        print(f"⚠️  重複データ: {duplicates} 行")
        contra_combined = contra_combined.drop_duplicates().reset_index(drop=True)
        print(f"✓ 重複除去後: {len(contra_combined)} 行")
    else:
        print("✓ 重複データなし")

    # グラフ構築
    if os.path.exists(aba_file_path) and 'node_id' not in pd.read_csv(aba_file_path, nrows=1).columns:
        # 通常フロー: 元ABAデータ（全Topic） + 対立CSVからグラフ生成
        cols = ['ReviewID', 'Title', 'Topic', 'Pos/Neg', 'Claim', 'Head',
        'Body 1', 'Body 2', 'Body 3', 'Body 4', 'Body 5', 'Body 6', 'Body 7',
        'Body 8', 'Body 9', 'Body 10', 'Body 11', 'Body 12', 'Body 13',
        'Body 14', 'Body 15', 'Cont. Body 1', 'Cont. Body 2', 'Cont. Body 3',
        'Cont. Body 4', 'Cont. Body 5', 'Cont. Body 6', 'Cont. Body 7', 
        'Cont. Body 8', 'Cont. Body 9', 'Cont. Body 10', 'Cont. Body 11',
        'Cont. Body 12', 'Cont. Body 13', 'Cont. Body 14', 'Cont. Body 15',]
        aba = pd.read_csv(aba_file_path, usecols=cols)
        # Topicによるフィルタは行わず、Room/Staffを含む全行を対象
        aba_all = aba
        aba_graph_room_staff = build_aba_graph(aba_all, contra_combined)
    else:
        # フォールバック: 対立CSVのみから最小限のグラフ（attackエッジのみ）を構築
        print(f"⚠️ 元ABAデータが見つかりませんでした（{aba_file_path}）。対立CSVのみから最小限のグラフを構築します。")
        g = nx.DiGraph()
        has_is_contrary = 'isContrary' in contra_combined.columns
        '''for _, r in contra_combined.iterrows():
            a = r.get('Assumption')
            p = r.get('Proposition')
            if pd.isna(a) or pd.isna(p):
                continue
            is_contrary = bool(r['isContrary']) if has_is_contrary else True
            if not is_contrary:
                continue
            g.add_node(a)
            g.add_node(p)
            g.add_edge(p, a, relation="attack")'''
        for _, r in contra_combined.iterrows():
            a = r.get('head')
            p = r.get('tail')
            if pd.isna(a) or pd.isna(p):
                continue
            relation = str(r.get('relation', '')).strip()
            g.add_node(a)
            g.add_node(p)
            if relation == 'CONTRARY_TO':
                g.add_edge(p, a, relation="attack")
            elif relation == 'NOT_CONTRARY':
                g.add_edge(p, a, relation="not_contrary")
        # Also load support (inference) edges
        support_path = os.path.join(project_root, "data/output/support_edges.csv")
        if os.path.exists(support_path):
            sup_df = pd.read_csv(support_path)
            for _, r in sup_df.iterrows():
                h = r.get('head')
                t = r.get('tail')
                if pd.isna(h) or pd.isna(t):
                    continue
                g.add_node(h)
                g.add_node(t)
                g.add_edge(h, t, relation="support")
            print(f"  Support edges added: {len(sup_df)} rows")
        print(f"  構築結果: ノード数={g.number_of_nodes()}, attackエッジ数={g.number_of_edges()}")
        aba_graph_room_staff = g

    # グラフ保存（統合データを反映したファイル名）
    #saved_file = save_graph_pickle(aba_graph_room_staff, "aba_graph_room_staff_combined")
    saved_file = save_graph_pickle(aba_graph_room_staff, "aba_graph_my_dataset_combined")
    print(f"\n保存完了: {saved_file}")
    print(f"📈 統合されたRoom/Staffのcontraデータを使用したグラフが保存されました")
