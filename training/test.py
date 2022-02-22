from transformers import PretrainedConfig, AutoModelForCausalLM, AutoTokenizer, GPT2Tokenizer

ckpt_path = 'comet-distill'
model = AutoModelForCausalLM.from_pretrained(ckpt_path)
tokenizer_path = 'comet-distill-tokenizer'
tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)