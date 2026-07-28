import config
import torch


from chunking import chunk_text



def score_chunk_pairs(model, resume, jd):
    resume_chunks = chunk_text(model,resume,chunk_size=config.RESUME_CHUNK_SIZE,overlap=config.RESUME_OVERLAP,max_chunks=config.MAX_RESUME_CHUNKS)
    jd_chunks = chunk_text(model,jd,chunk_size=config.JD_CHUNK_SIZE,overlap=config.JD_OVERLAP,max_chunks=config.MAX_JD_CHUNKS)
    
    pairs=[]
    
    for r_chunk in resume_chunks:
        for jd_chunk in jd_chunks:
            pairs.append((r_chunk, jd_chunk))
    
    return pairs


def top_chunk_score(model, resume, jd,k):
    pairs = score_chunk_pairs(model, resume, jd)
    
    resume_chunks = [pair[0] for pair in pairs]
    jd_chunks = [pair[1] for pair in pairs]
    
    inputs=model.tokenizer(resume_chunks, jd_chunks,padding=True,truncation=True,return_tensors="pt")

    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(config.device)

    
    output=model.model(**inputs)
    scores=output.logits.squeeze(-1)
    
    k=min(k, len(scores))
    top_k_scores= torch.topk(scores, k).values
    return top_k_scores.mean()


def compute_batch_scores(model, resumes, jds):
    scores=[]
    for resume, jd in zip(resumes, jds):
        score=top_chunk_score(model, resume, jd, k=3)
        scores.append(score)
    return torch.stack(scores)