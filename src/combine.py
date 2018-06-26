import os, sys
import codecs

directory, f = sys.argv[1], sys.argv[2]
with open(f, 'w+') as outfile:
	for root, dirs, files in os.walk(directory):
		for file in files:
			print("Writing file " + str(root) + '/' + str(file))
			path = root + '/' + file
			for line in codecs.open(path, 'r', encoding='utf-8', errors='ignore'):
				if '<doc' in line:
					pass
				else:
					outfile.write(line + ' ')