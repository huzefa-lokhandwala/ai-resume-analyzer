import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Read input files
with open("resume.txt", "r", encoding="utf-8") as file:
    resume_text = file.read()

with open("job_description.txt", "r", encoding="utf-8") as file:
    job_text = file.read()

# Simple text preprocessing
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

resume_clean = clean_text(resume_text)
job_clean = clean_text(job_text)

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform([resume_clean, job_clean])

similarity_score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

resume_words = set(resume_clean.split())
job_words = set(job_clean.split())

matched_keywords = resume_words.intersection(job_words)
missing_keywords = job_words.difference(resume_words)

print("Resume Match Score:", round(similarity_score * 100, 2), "%")
print("\nMatched Keywords:")
print(", ".join(list(matched_keywords)[:15]))

print("\nMissing Keywords:")
print(", ".join(list(missing_keywords)[:15]))
