import time
from .agents.legacy.call_to_action_agent import CallToActionAgent
from .agents.framing_agent import FramingAgent
from .agents.legacy.otherness_agent import OthernessAgent
from .agents.intent_agent import IntentAgent
from .agents.classification_agent import ClassificationAgent
from datetime import datetime

import pandas as pd

### Configuration ###
AVAILABLE_MODELS = [
    # "mistral",
    # "mistral-nemo",
    # "mistral-small",
    # "qwen3:8b",
    # "qwen3:14b",
    "gemma3:12b",
    "gemma3:27b"
]


mode = "no-context"
context = "whatever, irrelevant"

for model in AVAILABLE_MODELS:
    ### Data Loading ###
    df = pd.read_csv("data/testdata/processed_VideoCommentsThreatCorpus.csv")
    grouped_messages = pd.read_csv("data/testdata/grouped_processed_VideoCommentsThreatCorpus.csv")

    ### Logging Setup ###
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    start_time = time.time()

    ### Agent Initialization ###
    otherness_agent = OthernessAgent(model)
    framing_agent = FramingAgent(model)
    intent_agent = IntentAgent(model)
    classification_agent = ClassificationAgent(model)
    call_to_action_agent = CallToActionAgent(model)

    ### Output Storage Initialization ###
    collected_data = pd.DataFrame(columns=[
        'document_id',
        'num_posts_in_conversation',
        'conversation_length',
        'violence_label',
        'intent_label',
        'call_to_action',
        'flagged_issues',
        'row_duration_sec',
        'framing_style', 'framing_tool'
    ])


    for index,row in grouped_messages.iterrows():
        row_start_time = time.time()
        content_with_ids = df
        raw_content = row['content']
        content_list = raw_content.split("###---###")
        content = "\nNew message:\n".join(content_list)
        num_posts_in_conversation = row['num_posts']
        conversation_length = row['content_length']
        list_of_ids:list = row['id'].split(", ")

        print(f"Processing row {index + 1} of {len(grouped_messages)}...")

        for index,post in df[df['id'].isin(list_of_ids)].iterrows():
            specific_post_content = post['content']
            
            try:
                specific_post_framing = framing_agent.__call__(specific_post_content, context,mode=mode)
                specific_post_intent = intent_agent.__call__(specific_post_content, specific_post_framing, context,mode=mode)
                specific_post_call_to_action = specific_post_intent['call_to_action']
                specific_post_intent_of_violence = specific_post_intent['intent_of_violence']
                specific_post_classification = classification_agent.__call__(specific_post_content, framing_style = specific_post_framing['framingStyle'], framing_tool = specific_post_framing['framingTool'], intent_of_violence=specific_post_intent_of_violence, call_to_action=specific_post_call_to_action, context=context,mode=mode)

                row_duration_sec = time.time() - row_start_time

                new_row = {'document_id': post['id'], 'num_posts_in_conversation': num_posts_in_conversation, 
                'conversation_length': conversation_length,  
                'violence_label': specific_post_classification['label'], 'intent_label': specific_post_intent_of_violence, 
                'call_to_action': specific_post_call_to_action, 'flagged_issues': specific_post_classification['flagged_issues'],
                'row_duration_sec': row_duration_sec, 'framing_style': specific_post_framing['framingStyle'], 'framing_tool': specific_post_framing['framingTool']}

            except Exception as e:
                print(f"Error processing post {post['id']}: {e}")
                row_duration_sec = time.time() - row_start_time
                new_row = {'document_id': post['id'], 'num_posts_in_conversation': num_posts_in_conversation, 
                'conversation_length': conversation_length,  
                'violence_label': -1, 'intent_label': "None", 
                'call_to_action': "None", 'flagged_issues': "None",
                'row_duration_sec': row_duration_sec}

            # Convert new_row to a DataFrame and concatenate with the existing DataFrame
            new_row_df = pd.DataFrame([new_row])
            collected_data = pd.concat([collected_data, new_row_df], ignore_index=True)

            
    ### COLLECTION OF DATA ###

    collected_data.to_csv(f"data/testdata/test_results_from_idun/no_context/no_context_{model}_{timestamp}.csv",index=False)
