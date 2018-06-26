import json
import sys
from urllib.parse import unquote
import codecs

def reader(file):
	with open(file, 'r') as infile:
		for line in infile:
			yield line

def main():
	file, dictionary = sys.argv[1], sys.argv[2]
	count = 0
	with open(file, 'r') as infile:
		with open(dictionary, 'w+') as outfile:
			for line in infile:
			# for line in codecs.open(file, 'r', encoding='utf-8', errors='ignore'):
				description = {}
				if line.startswith('<url>'):
					entity = unquote(line)
					entity = entity.replace('https://en.wikipedia.org/wiki', '').lstrip('<url>').replace('</url>', '').strip()
					# print(entity)
				if line.startswith('<abstract>'):
					abstract = line.lstrip('<abstract>').replace('</abstract>', '').replace('(', '').replace(')', '').strip()
					# print(abstract)
					count += 1
					print("Total abstracts : " + str(count), end='\r')
					description[entity] = abstract
					json.dump(description, outfile)
					outfile.write('\n')
	print("Total abstracts : " + str(count))
	
if __name__ == '__main__':
	main()