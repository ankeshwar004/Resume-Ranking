import config
import tqdm
import torch
import torch.nn.functional as F

from src.chunking import chunk_text
from src.metric import model_evaluation


def cosent_loss(resume_emb, jd_emb, labels):
    similarity = F.cosine_similarity(resume_emb, jd_emb)

    similarity = similarity*10

    diff = []

    for i in range(len(labels)):
        for j in range(len(labels)):
            if labels[i] < labels[j]:
                diff.append(similarity[i] - similarity[j])

    if len(diff) == 0:
        return similarity.sum() * 0

    diff = torch.stack(diff)

    diff = torch.cat([torch.zeros(1, device=diff.device),diff])
    loss = torch.logsumexp(diff, dim=0)

    return loss

def encode_chunks(model,chunks):
    inputs = model.tokenize(chunks,padding=True,truncation=True,return_tensors="pt")
    
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(config.device)
            
    output = model(inputs)
    return output["sentence_embedding"]


def get_jd_embedding(model,jd):
    chunks=chunk_text(model,jd,chunk_size=config.COMMON_CHUNK_SIZE,overlap=config.COMMON_OVERLAP,max_chunks=config.MAX_COMMON_CHUNKS)
    embeddings=encode_chunks(model,chunks)
    embeddings=embeddings.mean(dim=0)
    return F.normalize(embeddings, p=2, dim=0)


def get_resume_embedding(model,resume,jd_embedding=None):
    chunks=chunk_text(model,resume,chunk_size=config.COMMON_CHUNK_SIZE,overlap=config.COMMON_OVERLAP,max_chunks=config.MAX_COMMON_CHUNKS)
    embeddings=encode_chunks(model,chunks)
    embeddings=F.normalize(embeddings, p=2, dim=1)
    
    if jd_embedding==None:
        embeddings=embeddings.mean(dim=0)
        return F.normalize(embeddings, p=2, dim=0)
    
    similarity=torch.matmul(embeddings,jd_embedding)
    best_idx=torch.argmax(similarity)
    return embeddings[best_idx]


def compute_batch_embeddings(model,resumes,jds):
    resume_embs=[]
    jd_embs=[]
    
    for resume,jd in zip(resumes,jds):
        jd_emb=get_jd_embedding(model,jd)
        resume_emb=get_resume_embedding(model,resume,jd_emb)
        
        resume_embs.append(resume_emb)
        jd_embs.append(jd_emb)
        
    resume_embs=torch.stack(resume_embs)
    jd_embs=torch.stack(jd_embs)
    
    return resume_embs, jd_embs


