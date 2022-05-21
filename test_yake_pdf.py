import yake
import pdfplumber


text = ""

with pdfplumber.open("./data/test1.pdf") as pdf:
    totalPages = len(pdf.pages)
    for i in range(0 , totalPages):
        page_obj = pdf.pages[i]
        text = text + page_obj.extract_text()

kw_extractor = yake.KeywordExtractor(lan="id",n=3 ,stopwords="id",top=50)
kata_kunci = kw_extractor.extract_keywords(text)

for i in kata_kunci:
    #print(i)
    kw = next(iter(i))
    print(kw)
