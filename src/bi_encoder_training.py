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
    inputs = {key: value.to(config.device) for key, value in inputs.items()}
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



def validate_bi_encoder(bi_encoder,val_df):
    bi_encoder.eval()
            
    with torch.no_grad():
            
        val_resume_embs, val_jd_embs=compute_batch_embeddings(bi_encoder,val_df['resume_text'].values,val_df['job_description_text'].values)
        
        scores=F.cosine_similarity(val_resume_embs,val_jd_embs,dim=1)
        scores=scores.cpu().numpy()
        
        metrics=model_evaluation(scores,val_df,"job_description_text")
        print("NDCG:", metrics["ndcg_val"])
        print("MAP:", metrics["map_score"])
        
        final_score=0.6*metrics["ndcg_val"]+0.3*metrics["map_score"]+0.1*metrics["mrr_score"]
        
        return final_score, metrics


def train_bi_encoder(bi_encoder,train_loader,val_df,optimizer,epochs,best_model_path,min_delta,patience):
    
    for epoch in range(epochs):
        
        print(f"Epoch {epoch+1}/{epochs}")
        bi_encoder.train()
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for resumes,jds,labels in progress_bar:
            optimizer.zero_grad()
            
            resume_embs, jd_embs=compute_batch_embeddings(bi_encoder,resumes,jds)
            
            labels=labels.to(config.device)
            loss=cosent_loss(resume_embs,jd_embs,labels)
            
            loss.backward()
            optimizer.step()
            
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            
            print("Final Loss:",loss.item())
            
            
        #Validation    
        
        final_score, metrics=validate_bi_encoder(bi_encoder,val_df)
            
        if final_score>best_score+min_delta:
            best_score=final_score
            bi_encoder.save(best_model_path)
            count=0
        else:
            count+=1

        if count==patience:
            print("Early stopping triggered.")
            break
                    
                
        