from openai import OpenAI
import sys
import argparse
import yaml

def query_openai(prompt):

    # load OpenAI API key
    with open('secrets.yaml', 'r') as file:
        openai_api_key = yaml.safe_load(file).get('OPENAI_API_KEY')
   
    client = OpenAI(
        api_key = openai_api_key
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages = [
                { "role": "system", "content": "You can only respond in bash code without syntax formatting." },
                { "role": "user", "content": prompt }
            ],
            logprobs=True
        )
        answer = response.choices[0].message.content.strip()
        return answer
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A command-line LLM tool")
    parser.add_argument("input", type=str, help="Input prompt for LLM")
    args = parser.parse_args()
    print(query_openai(args.input))
