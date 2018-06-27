import json
import sys
from urllib.parse import unquote
import codecs
import string, unicodedata

# def reader(file):
# 	with open(file, 'r') as infile:
# 		for line in infile:
# 			yield line

def main():
	"""
	The script processes input XML dump for all Wikipedia Abstracts and outputs a JSON file with the entities
	and their abstracts in key: value pairs.

	dictionary[entity] = abstract
	"""
	file, dictionary = sys.argv[1], sys.argv[2]
	count = 0
	table = str.maketrans('', '', string.punctuation)
	with open(file, 'r') as infile:
		with open(dictionary, 'w+') as outfile:
			for line in infile:
			# for line in codecs.open(file, 'r', encoding='utf-8', errors='ignore'):
				description = {}
				"""
				I am trying to remove the url from the doc to get the title and then cleaning the abstract for
				that title.
				"""
				if line.startswith('<url>'):
					entity = unquote(line)
					entity = "resource" + entity.replace('https://en.wikipedia.org/wiki', '').lstrip('<url>').replace('</url>', '').strip()
					# print(entity)
				if line.startswith('<abstract>'):
					abstract = line.lstrip('<abstract>').replace('</abstract>', '').replace('(', '').replace(')', '').strip()
					abstract = abstract.translate(table)
					abstract = ' '.join(abstract.split())
					# print(abstract)
					count += 1
					print("Total abstracts : " + str(count), end='\r')
					description[entity] = abstract
					json.dump(description, outfile)
					outfile.write('\n')
	print("Total abstracts : " + str(count))
	
if __name__ == '__main__':
	main()