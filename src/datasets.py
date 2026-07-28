from torch.utils.data import Dataset

class ResumeJDDataset(Dataset):
    def __init__(self,resumes,jds,labels):
        self.resumes=resumes
        self.jds=jds
        self.labels=labels
    
    def __len__(self):
        return len(self.resumes)
    
    def __getitem__(self, idx):
        return self.resumes[idx], self.jds[idx], self.labels[idx]
    