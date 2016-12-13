import json
out_file = open('cat_words.json', 'w')
pairs = []
counter = -1
prev = ''
with open('final_val.txt', 'r') as in_file:
    for line in in_file:
        line2 = line.split(' ')
        line3 = line2[0].split('/')
        curr = line3[0]
        if(prev != curr):
            prev = curr
            counter += 1
            temp = {}
            temp[counter] = line3[0]
            pairs.append(temp)
json.dump(pairs,out_file)
