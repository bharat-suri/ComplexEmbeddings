import argparse
import re

def main(args):
	"""
	I wanted to list all the titles present in the aricles dump. Currently that task is being performed through
	the script description.py
	"""

	pattern1 = re.compile(r"\[\[(\w+)\]\]")			# All patterns of the form [[article_title]]
	pattern2 = re.compile(r"\{\{(\w+)\}\}")			# All patterns of the form {{article_title}}
	entities = set()
	with open(args.input, 'r') as inFile:
		for line in inFile:
			if '<text' in line:						# Trying to parse the XML dump through the tag structure.
				try:
					entity = str(pattern1.search(line).group(1))
					entities.add(entity)
					entity = str(pattern2.search(line).group(1))
					entities.add(entity)
				except:
					continue
	with open(args.output, 'w') as outFile:
		for e in entities:
			outFile.write(e+"\n")

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--input", default="data/enwik9",
							help="Input Wiki XML dump")
	parser.add_argument("-o", "--output", default="data/entities",
							help="Output file with all the entities")
	args = parser.parse_args()
	main(args)