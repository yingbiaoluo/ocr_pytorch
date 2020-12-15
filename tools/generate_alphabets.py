

char_path = '/home/lyb/datasets/OCR/Sythetic_Chinese_Character_Dataset/train.txt'
with open(char_path) as f:
    content1 = f.read()

char_path = '/home/lyb/datasets/OCR/Sythetic_Chinese_Character_Dataset/test.txt'
with open(char_path) as f:
    content2 = f.read()

content = content1 + content2
char_set = sorted(list(set(content)))
print(char_set)

char_set.remove('\n')
char_set.remove(' ')

char_set = ''.join(char_set)
print(char_set)

path = '/home/lyb/ocr/text_det_reg/config/data/alphabets1.txt'
with open(path, 'w') as f:
    f.write(char_set)