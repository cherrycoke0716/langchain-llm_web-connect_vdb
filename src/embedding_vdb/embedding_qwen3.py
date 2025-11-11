from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
import loader

model = SentenceTransformer("Qwen/Qwen3-Embedding-4B")

queries = [

]