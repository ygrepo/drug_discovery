import tqdm


def get_embeddings(df):
    embeddings1 = []
    embeddings2 = []
    # Use a small subset for speed (remove this line to run on full dataset)
    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            emb1 = embed_sequence(row["protein1"])
            emb2 = embed_sequence(row["protein2"])
            embeddings1.append(emb1)
            embeddings2.append(emb2)
        except Exception as e:
            print("Embedding error:", e)
    return embeddings1, embeddings2
