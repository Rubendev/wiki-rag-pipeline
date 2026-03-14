from pipeline import run_genai

while True:
    query = ""
    query = input("Question: ")
    answer = run_genai([query])
    print("Answer:", answer[0]['answer'])
    print()
