import yake
import pdfplumber
import csv


text = ""
text_key = ""
final_text = []

#header = ["test_id", "answer", "key_answer"]


with pdfplumber.open("./data/test1.pdf") as pdf:
    totalPages1 = len(pdf.pages)
    for i in range(0 , totalPages1):
        page_obj = pdf.pages[i]
        text = text + page_obj.extract_text()


with pdfplumber.open("./data/test2.pdf") as pdf2:
    totalPages = len(pdf.pages)
    for i in range(0 , totalPages):
        page_obj = pdf.pages[i]
        text = text + page_obj.extract_text()


#data = ["0", text, text_key]

#with open("./data/predictest1.csv", "w", encoding="UTF8", newline="") as file_train:
#    writer = csv.writer(file_train)

#    writer.writerow(header)

#    writer.writerow(data)


kw_extractor = yake.KeywordExtractor(lan="id",n=3 ,stopwords="id",top=50)
kata_kunci = kw_extractor.extract_keywords(text)

for i in kata_kunci:
    #print(i)
    kw = next(iter(i))
    kw = str(kw)
    final_text.append(kw)
    

print (final_text)
