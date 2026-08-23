import config



def chunk_text(model,text, chunk_size, overlap, max_chunks):
    tokenizer=model.tokenizer
    ids=tokenizer.encode(text, add_special_tokens=False, verbose=False)
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


def get_resume_chunks(model, resume,resume_chunk_map):

    if resume not in resume_chunk_map:
        resume_chunk_map[resume] = chunk_text(model,resume,chunk_size=config.RESUME_CHUNK_SIZE,
                                              overlap=config.RESUME_OVERLAP,max_chunks=config.MAX_RESUME_CHUNKS)

    return resume_chunk_map[resume]


def get_jd_chunks(model, jd,jd_chunk_map):

    if jd not in jd_chunk_map:
        jd_chunk_map[jd] = chunk_text(model,jd,chunk_size=config.JD_CHUNK_SIZE,
                                      overlap=config.JD_OVERLAP,max_chunks=config.MAX_JD_CHUNKS)

    return jd_chunk_map[jd]
