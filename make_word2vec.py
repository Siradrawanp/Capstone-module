import gensim
import logging
import pandas as pd


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def extract_answer():
    df1 = pd.read_csv()
    df2 = pd.read_csv()

    for dataset in [df1, df2]:
        for i, row in dataset.iterrows():
            if  i != 0 and i % 1000 ==0:
                logging.info("read {0} sentences".format(i))

            if row['answer']:
                yield gensim.utils.simple_preprocess(row['answer'])
            if row['key_answer']:
                yield gensim.utils.simple_preprocess(row['key_answer'])

documents = list(extract_answer())
logging.info("done reading data file")

model = gensim.models.Word2Vec(documents, size=100)
model.train(documents, total_examples=len(documents), epochs=10)
model.save("./data/answer_key_answer_pairs.w2v")