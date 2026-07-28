import config



def chunk_text(model,text, chunk_size, overlap, max_chunks):
    tokenizer=model.tokenizer
    ids=tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= chunk_size:
        return [text]

    chunks = []
    steps = chunk_size - overlap
    
    for i in range(0, len(ids), steps):
        chunk_ids = ids[i:i + chunk_size]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text)

        if len(chunks) >= max_chunks:
            break
    return chunks

