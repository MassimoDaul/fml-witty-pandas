'''
Export trained GNN embeddings to the `andrew` column in pgvector.

Usage:
    python embed.py
    python embed.py --checkpoint path/to/checkpoint.pt
'''

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from database.utils import get_connection, drop_ivf_indexes, upsert_embeddings, build_ivf_indexes
from train import build_graph, PaperGNN, CHECKPOINT_PATH, DEVICE


def embed(checkpoint_path: str = CHECKPOINT_PATH):
    print("Loading graph data...")
    data, _, _, corpus_ids, _ = build_graph()
    data = data.to(DEVICE)

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model = PaperGNN(data.metadata()).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(
        f"  epoch={checkpoint['epoch']} "
        f"train_loss={checkpoint['train_loss']:.4f} "
        f"val_loss={checkpoint['val_loss']:.4f}"
    )

    print("Running inference...")
    with torch.no_grad():
        z = model(data).cpu().numpy()  # (N, 128)

    print(f"Exporting {len(corpus_ids)} embeddings to 'andrew' column...")
    conn = get_connection()
    drop_ivf_indexes(conn, 'andrew')
    upsert_embeddings(conn, 'andrew', list(zip(corpus_ids, z)))
    build_ivf_indexes(conn, 'andrew')
    conn.close()

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=CHECKPOINT_PATH)
    args = parser.parse_args()
    embed(args.checkpoint)
