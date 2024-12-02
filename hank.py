import sys
import argparse
import yaml
from openai import OpenAI

def finetuned_query(prompt):
    with open('secrets.yaml', 'r') as file:
        secrets_dict = yaml.safe_load(file)
        openai_api_key = secrets_dict.get('OPENAI_API_KEY')
        finetuned_model_id = secrets_dict.get('FINETUNED_MODEL_ID')

    client = OpenAI(api_key = openai_api_key)

    try:
        response = client.chat.completions.create(
                model=finetuned_model_id,
                messages = [
                    { "role": "system", "content": "You can only respond in bash code without syntax formatting." },
                    { "role": "user", "content": prompt }
                ]
            )
        answer = response.choices[0].message.content.strip()
        return answer
    except Exception as e:
        return f'Error: {e}'


# function to faciliate completion to gpt-4o-mini model
def zeroshot_query(prompt):

    # load OpenAI API key
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A command-line LLM tool")
    parser.add_argument('input', type=str, help="Input prompt for LLM")
    parser.add_argument('-z', '--zero-shot', action='store_true', help='Send prompt to zero-shot model')
    parser.add_argument('-f', '--fine-tuned', action='store_true', help='Send prompt to fine-tuned model')
    args = parser.parse_args()
    if args.zero_shot:
        print(zeroshot_query(args.input))
    elif args.fine_tuned:
        print(finetuned_query(args.input))

