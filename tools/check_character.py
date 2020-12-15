import os


if __name__ == '__main__':
    char_path = '/home/lyb/datasets/OCR/Sythetic_Chinese_Character_Dataset/train.txt'
    with open(char_path) as f:
        content1 = f.read()

    char_path = '/home/lyb/datasets/OCR/Sythetic_Chinese_Character_Dataset/test.txt'
    with open(char_path) as f:
        content2 = f.read()

    char_path = '/home/lyb/ocr/text_det_reg/data/images_sentences/sentences_label.txt'
    with open(char_path) as f:
        content3 = f.read().splitlines()

    content = content1 + content2
    char_set = set(content)

    f_chars = ['；', '：', '？', '∥', '査', '岀', '\t', '項', 'Ⅹ', '∨', '｝', 'μ', 'ⅴ', 'Ⅴ', '∧',
               '釆', '！', '吋', '屮', '裎', '沖', '谝', '內', '仼', '別', 'ν', '趙', '仝', '鍵', 'ⅹ',
               '狳', '晩', '仵', '冋', '題', '攴', '戋']
    t_chars = [';',  ':', '?', '//', '查', '出', '',  '项', 'X', 'v', '}',  'u', 'v', 'V', '^',
               '采', '!', '时', '中', '程', '冲', '遍', '内', '任', '别', 'v', '趟', '全', '键', 'x',
               '除', '晚', '件', '同', '题', '支', '划']

    # print('content3:', content3)
    content3_mod = []
    for line in content3:
        line_mod = line

        for char in line:
            if char not in char_set:
                # print(char, '\t',line)
                if char in f_chars:
                    index = f_chars.index(char)
                    t_char = t_chars[index]
                    line_mod = line_mod.replace(char, t_char)

        content3_mod.append(line_mod)

    char_set1 = set(''.join(content3))

    print(char_set)
    print(char_set1)
    print(len(char_set), len(char_set1))

    for line in content3_mod:
        for char in line:
            if char not in char_set:
                print(char, '\t', line)

    # char_path = '/home/lyb/ocr/text_det_reg/data/images_sentences/sentences_label1.txt'
    # with open(char_path, 'w') as f:
    #     for line in content3_mod:
    #         f.write(line+'\n')