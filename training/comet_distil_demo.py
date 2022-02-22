from distil_comet import DistilComet
import os

storage_path = '/srv/share2/ahavrilla3'


model_ckpt = 'light_package/comet-distill'
tokenizer_path = 'light_package/comet-distill-tokenizer'

model = DistilComet(os.path.join(storage_path, model_ckpt), os.path.join(storage_path, tokenizer_path))
model.eval()
query = [["PersonX eats an apple", 'xNeed']]
output = model.forward(query)
print(output)