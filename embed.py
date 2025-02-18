from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

def get_embedding(input):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(input)
    pca = PCA(n_components=1)
    pca.fit(embeddings)
    single_values = pca.transform(embeddings)
    results = [value[0] for value in single_values]
    print(results)
    return results

if __name__ == "__main__":
    sentence1 = input("Sentence 1: ")
    sentence2 = input("Sentence 2: ")
    input = [sentence1, sentence2]
    get_embedding(input)
