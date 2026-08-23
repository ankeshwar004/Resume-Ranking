import os
import torch
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = os.path.join(BASE_DIR, "data")

RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
CLEANED_DATA_DIR = os.path.join(DATA_DIR, "cleaned")
PREPROCESSED_DATA_DIR = os.path.join(DATA_DIR, "preprocessed")

SKILLS_RAW_DIR = os.path.join(DATA_DIR, "skills", "raw")
SKILLS_PROCESSED_DIR = os.path.join(DATA_DIR, "skills", "processed")

HF_DATASET_DIR = os.path.join(RAW_DATA_DIR, "resume-job-description-fit")
RAW_TEST_CSV = os.path.join(RAW_DATA_DIR, "test.csv")

SKILLS_CSV = os.path.join(SKILLS_RAW_DIR, "JobsDatasetProcessed.csv")
ALL_SKILLS_PKL = os.path.join(SKILLS_PROCESSED_DIR, "all_skills.pkl")

MODELS_DIR = os.path.join(BASE_DIR, "models")

BASELINE_MODEL_DIR = os.path.join(MODELS_DIR, "baseline")
CHUNKED_MODEL_DIR = os.path.join(MODELS_DIR, "chunked")
HD_MODEL_DIR = os.path.join(MODELS_DIR, "hard_negative")

RESULTS_DIR = os.path.join(BASE_DIR, "results")

device = "cuda" if torch.cuda.is_available() else "cpu"

label_to_score={0:0,1:0.5,2:1.0}

BATCH_SIZE=64
BASELINE_BATCH_SIZE=64
CHUNK_BATCH_SIZE=8

JD_CHUNK_SIZE = 160
RESUME_CHUNK_SIZE = 300

JD_OVERLAP = 30
RESUME_OVERLAP = 50

MAX_JD_CHUNKS = 4
MAX_RESUME_CHUNKS = 6


COMMON_CHUNK_SIZE=200
COMMON_OVERLAP=50
MAX_COMMON_CHUNKS=10
