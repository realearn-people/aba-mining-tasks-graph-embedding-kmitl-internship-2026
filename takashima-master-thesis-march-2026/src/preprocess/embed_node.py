import torch
import pickle
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

def _get_device():
    """Use GPU only when nvidia-smi reports >= 1 GiB free (no CUDA context needed)."""
    import subprocess
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
            timeout=5
        )
        free_mib = int(out.decode().strip().split('\n')[0])
        if free_mib >= 1024:
            return torch.device('cuda')
    except Exception:
        pass
    return torch.device('cpu')

def generate_bert_embeddings(nodes, model_name='google-bert/bert-base-uncased'):
    """BERTエンベディングを生成"""
    device = _get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    node_embeddings = {}
    with torch.no_grad():
        for node in tqdm(nodes, desc="BERTエンベディング生成"):
            inputs = tokenizer(node, return_tensors='pt', padding=True, truncation=True, max_length=256)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
            node_embeddings[node] = embedding
    
    return node_embeddings

if __name__ == "__main__":
    # データの読み込み
    filepath = 'data/output/inference_graph_room.pkl'
    with open(filepath, 'rb') as f:
        inference_graph = pickle.load(f)

    # ノードの取得
    nodes = list(inference_graph.nodes())

    # エンベディングの生成
    node_embeddings = generate_bert_embeddings(nodes)

    # エンベディングの保存
    with open('data/output/node_embeddings_room.pkl', 'wb') as f:
        pickle.dump(node_embeddings, f)