# -*- coding: utf-8 -*

def get_cangjie(char):
	cns_char = 'cns_char.txt'
	char_dict = dict()
	with open(cns_char, 'rb') as f:
		for line in f.readlines():
			line = line.decode().strip()
			split = line.split('\t')
			if len(split) == 2:
				char_dict[split[1]] = split[0]

	cns = char_dict.get(char)

	cns_cangjie = 'CNS_cangjie.txt'
	cangjie_dict = dict()
	with open(cns_cangjie, 'rb') as f:
		for line in f.readlines():
			line = line.decode().strip()
			split = line.split('\t')
			if len(split) == 2:
				cangjie_dict[split[0]] = split[1]

	return  cangjie_dict.get(cns)


def get_component(char):
	cns_char = 'cns_char.txt'
	char_dict = dict()
	with open(cns_char, 'rb') as f:
		for line in f.readlines():
			line = line.decode().strip()
			split = line.split('\t')
			if len(split) == 2:
				char_dict[split[1]] = split[0]

	cns = char_dict.get(char)

	cns_component = 'CNS_component.txt'
	component_dict = dict()
	with open(cns_component, 'rb') as f:
		for line in f.readlines():
			line = line.decode().strip()
			split = line.split('\t')
			if len(split) == 2:
				component = split[1].split(';')
				component_dict[split[0]] = component[0]

	return  component_dict.get(cns)


