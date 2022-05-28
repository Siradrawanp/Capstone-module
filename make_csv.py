import pdfplumber
import csv


text_ans = ""
text_key = ""
final_text = []

# fieldname untuk jawaban, answer1 = mahasiswa, answer2 = kunci jawaban
fieldname = ['test_id', 'answer1', 'answer2']

# memuat dan membaca teks file jawaban
with pdfplumber.open("./data/test1.pdf") as pdf1:
    totalPages1 = len(pdf1.pages)
    for i in range(0 , totalPages1):
        page_obj = pdf1.pages[i]
        text_ans = text_ans + page_obj.extract_text()


# memuat dan membaca teks file kunci jawaban
with pdfplumber.open("./data/test2.pdf") as pdf2:
    totalPages2 = len(pdf2.pages)
    for i in range(0 , totalPages2):
        page_obj = pdf2.pages[i]
        text_key = text_key + page_obj.extract_text()

# menetapkan kolom dan baris teks jawaban
rows = [
    {'test_id': 0,
    'answer1': text_ans,
    'answer2': text_key}
]

# membuat file csv untuk prediksi
with open("./data/predictest1.csv", "w", encoding="UTF8", newline="") as file_train_test:
    writer = csv.DictWriter(file_train_test, fieldnames=fieldname)

    writer.writeheader()

    writer.writerows(rows)
