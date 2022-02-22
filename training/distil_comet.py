from typing import List
import torch
from transformers import PretrainedConfig, AutoModelForCausalLM, AutoTokenizer, GPT2Tokenizer
import json
import os


class DistilComet:
  def __init__(self, ckpt_path: str, tokenizer_path: str):
    self.model = AutoModelForCausalLM.from_pretrained(ckpt_path)
    self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    self.device =  'cuda' if torch.cuda.is_available() else 'cpu'
    self.model.to(self.device)

  def eval(self) -> None:
    self.model.eval()

  def forward(self, queries: List[List[str]]) -> List[str]:
    'Query expected to be of the form [head, relation]'
    formatted_queries = []
    for query in queries:
      head = query[0]
      rel = query[1]
      formatted_query = "{} {} [GEN]".format(head, rel)
      formatted_queries.append(formatted_query)
    print(formatted_queries)

    encoded_queries = self.tokenizer.batch_encode_plus(formatted_queries, return_tensors='pt')
    encoded_queries = encoded_queries.to(self.device)
    print(encoded_queries)
    results = self.model.generate(**encoded_queries, decode_method="beam", num_generate=5)
    decoded_results = self.tokenizer.batch_decode(results)
    return decoded_results