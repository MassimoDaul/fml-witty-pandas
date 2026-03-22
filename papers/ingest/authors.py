def collect_author_paper_pairs(records: list[dict]) -> list[tuple[str, str]]:
    """
    Extract (author_name, arxiv_id) pairs from a batch of parsed paper records.
    Each record must have 'arxiv_id' and optionally 'authors' (list[str] or None).
    """
    pairs = []
    for record in records:
        arxiv_id = record["arxiv_id"]
        for name in record.get("authors") or []:
            if name:
                pairs.append((name, arxiv_id))
    return pairs
