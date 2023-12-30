
from transformers import AutoTokenizer, AutoModel

# Use raw string literals for Windows paths
model_path = r"F:\ChatGLM-data\chatglm-6b-int4-slim"


tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, ignore_mismatched_sizes=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True, ignore_mismatched_sizes=True).cpu().float()
response, history = model.chat(tokenizer, "你好", history=[])
print(response)
response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
print(response)

