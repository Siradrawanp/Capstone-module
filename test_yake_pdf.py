import yake
import pdfplumber
import csv


text_ans = ""
text_key = ""
final_text = []

fieldname = ['test_id', 'answer', 'key_answer']


with pdfplumber.open("./data/test1.pdf") as pdf1:
    totalPages1 = len(pdf1.pages)
    for i in range(0 , totalPages1):
        page_obj = pdf1.pages[i]
        text_ans = text_ans + page_obj.extract_text()


with pdfplumber.open("./data/test2.pdf") as pdf2:
    totalPages2 = len(pdf2.pages)
    for i in range(0 , totalPages2):
        page_obj = pdf2.pages[i]
        text_key = text_key + page_obj.extract_text()

rows = [
    {'test_id': 0,
    'answer': text_ans,
    'key_answer': text_key}
]

with open("./data/predictest1.csv", "w", encoding="UTF8", newline="") as file_train_test:
    writer = csv.DictWriter(file_train_test, fieldnames=fieldname)

    writer.writeheader()

    writer.writerows(rows)

'''
kw_extractor = yake.KeywordExtractor(lan="id",n=3 ,stopwords="id",top=50)
kata_kunci = kw_extractor.extract_keywords(text_ans)
kunci_jawab = kw_extractor.extract_keywords(text_key)

for i in kata_kunci:
    #print(i)
    kw = next(iter(i))
    kw = str(kw)
    final_text.append(kw)
    

print (final_text)
'''