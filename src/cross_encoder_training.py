import config
import torch


from src.chunking import get_resume_chunks,get_jd_chunks



def score_chunk_pairs(model, resume, jd, resume_chunk_map={}, jd_chunk_map={}):
    resume_chunks = get_resume_chunks(model,resume,resume_chunk_map)
    jd_chunks = get_jd_chunks(model,jd,jd_chunk_map)
    
    pairs=[]
    
    for r_chunk in resume_chunks:
        for jd_chunk in jd_chunks:
            pairs.append((r_chunk, jd_chunk))
    
    return pairs


def top_chunk_score(model, resume, jd,top_k, resume_chunk_map={}, jd_chunk_map={}):
    pairs = score_chunk_pairs(model, resume, jd, resume_chunk_map, jd_chunk_map)
    
    resume_chunks = [pair[0] for pair in pairs]
    jd_chunks = [pair[1] for pair in pairs]
    
    inputs=model.tokenizer(resume_chunks, jd_chunks,padding=True,truncation=True,return_tensors="pt")

    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(config.device)

    
    output=model.model(**inputs)
    scores=output.logits.squeeze(-1)
    
    top_k=min(top_k, len(scores))
    top_k_scores= torch.topk(scores, top_k).values
    return top_k_scores.mean()


def compute_batch_scores(model, resumes, jds, resume_chunk_map={}, jd_chunk_map={}):
    scores=[]
    top_k=3
    for resume, jd in zip(resumes, jds):
        score=top_chunk_score(model, resume, jd, top_k, resume_chunk_map, jd_chunk_map)
        scores.append(score)
    return torch.stack(scores)