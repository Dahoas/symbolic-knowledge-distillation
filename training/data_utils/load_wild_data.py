import json
import os
#Only working with wild_chats data for now

def load_raw_data(data_path, filename):
	file_path = os.path.join(data_path, filename)
	with open (file_path, 'rb') as f:
		data = json.load(f)
	return data

#Formatted according to distil_comet input form ['HEAD', 'REL']
def convert_dialog_to_motivation_data(data):
	data_format_concat = []
	data_format_separate = []
	for datapoint in data:
		datapoint = datapoint['conv_info']
		#[Name, Persona]
		entities = {}
		entity_dialogs = {}
		for act in datapoint['acts']:
			name = act['id']
			if entities.get(name) is None:
				persona = act['task_data']['persona']
				entities[name] = persona
				entity_dialogs[name] = []
			entity_dialogs[name].append(act['text'])

		for entity in entities.keys():
			head = ""
			for entity_dialog in entity_dialogs[entity]:
				head += f'<dialog> {entity_dialog} '
			rel = "xWant"
			tail = entities[entity]
			data_format_concat.append([head, rel, tail])

		for entity in entities.keys():
			for entity_dialog in entity_dialogs[entity]:
				head = entity_dialog
				rel='xWant'
				tail = entities[entity]
				data_format_separate.append([head, rel, tail])

	return data_format_concat, data_format_separate

def explore_data(data):
	n = len(data)
	print(n)
	datapoint = data[1]
	print(datapoint.keys())
	#datapoint = datapoint['conv_info']
	print(datapoint.keys())
	print(json.dumps(datapoint, sort_keys=True, indent=4))


if __name__ == "__main__":
	data_path = '/mnt/raid/users/AlexH/test/light/LIGHT_RM/data/quests/wild_chats'
	filename = 'formatted_wild_completions.json'
	data = load_raw_data(data_path, filename)
	explore_data(data)
	data = data[0:2]
	#converted = convert_dialog_to_motivation_data(data)
	#print(converted)


