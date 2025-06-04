
from .agents.SinglePrompt_agent import SinglePromptAgent
from model_config import AVAILABLE_MODELS
from datetime import datetime
from pydantic import ValidationError
import pandas as pd
import time

AVAILABLE_MODELS = [
    # "mistral",
    # "mistral-nemo",
    # "mistral-small",
    # "qwen3:8b",
    # "qwen3:14b",
    "gemma3:12b",
    "gemma3:27b"
]
for model in AVAILABLE_MODELS:
    ### Data Loading 
    df = pd.read_csv("data/testdata/processed_VideoCommentsThreatCorpus.csv")
    grouped_messages = pd.read_csv("data/testdata/grouped_processed_VideoCommentsThreatCorpus.csv")

    ### Logging Setup 
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    start_time = time.time()

    ### Agent Initialization
    singlePrompt_agent = SinglePromptAgent(model)

    ### Output Storage Initialization
    collected_data = pd.DataFrame(columns=[
        'document_id',
        'violence_label',
        'row_duration_sec',
        'flagged_issues'
    ])


    print("starting singlePrompt_agent processing")

    for index, row in grouped_messages.iterrows():
        row_start_time = time.time()
        content = row['content']
        list_of_ids:list = row['id'].split(", ")
        content_list = content.split("###---###")   
        results = []

        for index,post in df[df['id'].isin(list_of_ids)].iterrows():
            content = post['content']
            try:
                result = singlePrompt_agent.__call__(content)
            except ValidationError as e:
                print(f"Validation error for content: {content}, Error: {e}")
                result = {'violent_label': None, 'flagged_issues': [1]}
            results.append(result)

        row_duration_sec = time.time() - row_start_time

        for flag in results:
            new_row = {'document_id':post['id'], 'violence_label': flag['violent_label'], 'flagged_issues': flag['flagged_issues'], 'row_duration_sec': row_duration_sec}
            new_row_df = pd.DataFrame([new_row])
            collected_data = pd.concat([collected_data, new_row_df], ignore_index=True)


    collected_data.to_csv(f"data/testdata/test_results_from_idun/solo/solo{model}_{timestamp}.csv", index=False)
    print("solo processing completed and results saved.")

