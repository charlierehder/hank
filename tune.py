import json
import yaml
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# allows for progress bars during pandas operations, makes it seem professional ya know
tqdm.pandas()

# get OpenAI API key
with open('secrets.yaml', 'r') as file:
    openai_api_key = yaml.safe_load(file).get('OPENAI_API_KEY')

# set up OpenAI client
client = OpenAI(
        api_key=openai_api_key, 
        project = 'proj_PakLyGei2rGL2mVvuq3ndooU'
)

df = pd.read_json('data/nl2cmd.json')
df = df.T
print(f'The shape of the input data is {df.shape}')

# helper function to define training validation set as a dictionary
def prep_training_data(item): 
    system_message = 'You can only respond in bash code with syntax formatting'
    return {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": item["invocation"]},
                {"role": "assistant", "content": item["cmd"]}
            ]
    }

# create formatted set of training data - starting with 70/30 split
print('Creating train/test split data...')
training_df = df.loc[0:int(len(df) * 0.7) - 1]
training_df = training_df.progress_apply(prep_training_data, axis=1).tolist()
print('Complete.')

# similarly for validation data
validation_df = df.loc[int(len(df) * 0.7):]
validation_df = validation_df.progress_apply(prep_training_data, axis=1).tolist()
print('Complete.')
print(f'Formatted {len(training_df)} training and {len(validation_df)} validation samples')

training_data_filename = 'training_data.jsonl'
validation_data_filename = 'validation_data.jsonl'

# write formatted data to output
print(f'Writing training data to {training_data_filename}...')
with open(training_data_filename, 'w') as train_file:
    for line in tqdm(training_df):
        train_file.write(json.dumps(line) + '\n')


# similarly for validation data
print(f'Writing validation data to {validation_data_filename}...')
with open(validation_data_filename, 'w') as valid_file:
    for line in tqdm(validation_df):
        valid_file.write(json.dumps(line) + '\n')


purpose = 'fine-tune'

# upload training file to OpenAI
print('Uploading training file to OpenAI')
with open(training_data_filename, 'rb') as train_file:
    response = client.files.create(file=train_file, purpose=purpose)
    training_file_id = response.id
print(f'Training file id: {training_file_id}')

# upload validation file to OpenAI
print('Uploading validation file to OpenAI')
with open(validation_data_filename, 'rb') as valid_file:
    response = client.files.create(file=valid_file, purpose=purpose)
    validation_file_id = response.id
print(f'Validation file id: {validation_file_id}')

# instantiate fine-tuning job
model = "gpt-4o-mini-2024-07-18"

response = client.fine_tuning.jobs.create(
    training_file=training_file_id,
    validation_file=validation_file_id,
    model=model,
    suffix="hank",
)

job_id = response.id

print('Checking file upload status')
while True:

    # wait 10 seconds between checks
    time.sleep(5)

    train_response = client.files.retrieve(training_file_id)
    valid_response = client.files.retrieve(validation_file_id)
    if valid_response.status in ('processed'): 
        print(f'Status: {valid_response.status}')
        break

print('File upload complete')


print('Starting fine-tuning job...')
print("Job ID:", response.id)

last_event = ""
while True:

    # print events to out if they've changed
    response = client.fine_tuning.jobs.list_events(job_id)
    events = response.data
    if events[0].message != last_event:
        print(events[0].message)
        last_event = events[0].message

    response = client.fine_tuning.jobs.retrieve(job_id)
    if response.status in ('succeeded', 'cancelled', 'failed'):
        print(f'Fine-tuned model_id: {response.fine_tuned_model}')
        break
 
    # wait 5 seconds between status calls
    time.sleep(5)


