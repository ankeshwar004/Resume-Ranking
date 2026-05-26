


def map_metric(group,relevant_label=1):

    group = group.sort_values('score', ascending=False)

    if (group['label'] >= relevant_label).sum() == 0:
        return None

    labels = group['label'].values
    precisions = []
    relevant_found = 0

    for rank, label in enumerate(labels, start=1):
        if label >= relevant_label:
            relevant_found += 1
            precision_at_k = relevant_found / rank
            precisions.append(precision_at_k)

    if len(precisions) == 0:
        return 0

    return np.mean(precisions)

def mrr_metric(group,relevant_label=1):
    
    if (group['label'] >= relevant_label).sum() == 0:
        return None
        
    group = group.sort_values('score', ascending=False)
    labels = group['label'].values

    for rank, label in enumerate(labels, start=1):
        if label >= relevant_label:
            return 1 / rank

    return 0

def corr_metric(group):
    if len(group['label'].unique())>1:
        corr,_=spearmanr(group['label'],group['score'])
        
        if not np.isnan(corr):
            return corr
    return None

    

def ndcg_metric(group):

    if(len(group)<2):
        return None
    
    true_scores = [group['label'].values]
    predicted_scores = [group['score'].values]

    return ndcg_score(true_scores,predicted_scores,k=min(10, len(group)))

def topk_metric(group, k=3):
    
    if 2 not in group['label'].values:
        return None
        
    group = group.sort_values('score',ascending=False)
    top_k_labels = group.head(k)['label'].values

    if (top_k_labels==2).any():
        return 1

    return 0


def evaluate_per_jd(df,score,col_name,metric_func):
    df=df.copy()
    df['score']=score
    
    scores=[]

    for jd, group in df.groupby(col_name):

        metric_val=metric_func(group)
        
        if metric_val is not None:
            scores.append(metric_val)

    return np.mean(scores) if scores else 0
    
    
def model_evaluation(scores,test_df,col_name='job_description_text'):

  spearman_score=evaluate_per_jd(test_df,scores,col_name,corr_metric)
    
  topk_score=evaluate_per_jd(test_df,scores,col_name,
                                lambda group: topk_metric(group,3))
  ndcg_val=evaluate_per_jd(test_df,scores,col_name,ndcg_metric)
    
  mrr_score=evaluate_per_jd(test_df,scores,col_name,
                               lambda group: mrr_metric(group,1))
    
  map_score=evaluate_per_jd(test_df,scores,col_name,
                             lambda group: map_metric(group,1))  

  print("Spearman:", spearman_score)
  print("Top-3 Accuracy:", topk_score)
  print("NDCG:", ndcg_val)
  print("MRR:", mrr_score)
  print("MAP:", map_score)


  return {
    "spearman_score":spearman_score,
    "topk_score":topk_score,
    "ndcg_val":ndcg_val, 
    "mrr_score":mrr_score,
    "map_score":map_score
  }
