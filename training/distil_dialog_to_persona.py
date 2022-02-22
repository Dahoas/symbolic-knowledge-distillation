from load_data import load_raw_data, convert_dialog_to_motivation_data
from distil_comet import DistilComet

data_path = '/mnt/raid/users/AlexH/test/light/LIGHT_RM/data/quests/wild_chats'
filename = 'formatted_wild_completions.json'
data = load_raw_data(data_path, filename)
formatted_data_concat, data_separate = convert_dialog_to_motivation_data(data)

model_ckpt = 'downloaded/comet-distill'
tokenizer_path = 'downloaded/comet-distill-tokenizer'

model = DistilComet(model_ckpt, tokenizer_path)
model.eval()

for datapoint in data_separate:
	input = [datapoint[:2]]
	output = model.forward(input)
	print(output)
	print(datapoint[2])
	break