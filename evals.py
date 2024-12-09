import yaml
import time
from openai import OpenAI
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def load_validation_data(file_path): 
    with open(file_path, "r") as f:
        return json.load(f)

def generate_response(prompt, model="gpt-4", max_tokens=100):
    with open('secrets.yaml', 'r') as file:
        openai_api_key = yaml.safe_load(file).get('OPENAI_API_KEY')

    client = OpenAI(api_key = openai_api_key)

    try:
        response = client.chat.completions.create(
            model='gpt-4o-mini-2024-07-18',
            messages = [
                { "role": "system", "content": "You can only respond in bash code without syntax formatting." },
                { "role": "user", "content": prompt }
            ]
        )
        answer = response.choices[0].message.content.strip()
        return answer
    except Exception as e:
        return f"Error: {e}"

def calculate_bleu_score(reference, candidate):

    reference_tokens = reference.split()
    candidate_tokens = candidate.split()
    return sentence_bleu(
        [reference_tokens],
        candidate_tokens,
        smoothing_function=SmoothingFunction().method1
    )

df = pd.read_json('data/nl2cmd.json')
df = df.T
validation_data = df.head(25)
print(f'The shape of the input data is {df.shape}')


with open('secrets.yaml', 'r') as file:
    secrets_dict = yaml.safe_load(file)
    finetuned_model_id = secrets_dict.get('FINETUNED_MODEL_ID')

bleu_scores = []    
for index, example in validation_data.iterrows():
    prompt = example["invocation"]
    reference = example["cmd"]
    candidate = generate_response(prompt, model=finetuned_model_id)
    bleu_score = calculate_bleu_score(reference, candidate)
    bleu_scores.append(bleu_score)
    print("----------")
    print(f"Prompt: {prompt}")
    print(f"Reference: {reference}")
    print(f"Candidate: {candidate}")
    print(f"BLEU Score: {bleu_score}")
    time.sleep(2)
    
avg_bleu = sum(bleu_scores) / len(bleu_scores)
print(f"Average BLEU Score: {avg_bleu}")
