
from src.agents.context_agent import ContextAgent
from src.agents.framing_agent import FramingAgent
from .agents.SinglePrompt_agent import SinglePromptAgent

from datetime import datetime
from pydantic import ValidationError
import pandas as pd
import time

AVAILABLE_MODELS = [
    "mistral",
    "mistral-nemo",
    "mistral-small",
    "qwen3:8b",
    "qwen3:14b",
    "gemma3:12b",
]

### Configuration 
models = AVAILABLE_MODELS  # mistral
pipes= ["neighbor","context","neighbor"]

### Data Loading 
df = pd.read_csv("data/testdata/processed_VideoCommentsThreatCorpus.csv")
grouped_messages = pd.read_csv("data/testdata/grouped_processed_VideoCommentsThreatCorpus.csv")

for model in models:
    framing_agent = FramingAgent(model)
    for pipe in pipes:

        ### Output Storage Initialization
        collected_data = pd.DataFrame(columns=[
            'document_id',
            'framing_style',
            'framing_tool',
            'pipe',
            'model'
        ])

        print(f"starting {pipe,model} processing")

        for index, row in grouped_messages.iterrows():
            
           
            list_of_ids:list = row['id'].split(", ")
            context = "whatever, irrelevant"

            for index,post in df[df['id'].isin(list_of_ids)].iterrows():
                content = post['content']

                if(pipe=="context"):
                    context_agent = ContextAgent(model)
                    raw_content = row['content']
                    content_list = raw_content.split("###---###")
                    content_for_context = "\nNew message:\n".join(content_list)
                    context = context_agent.__call__(content_for_context)
                    mode="context"

                elif(pipe=="neighbor"):
                    mode="neighbor"
                    neighbors_window = 1  # Number of neighboring posts before and after
                    post_id = post['id']
                    post_index = list_of_ids.index(post_id)

                    # Get context messages before the current post
                    context_before = "\n".join(
                        df[df['id'] == list_of_ids[j]]['content'].values[0]
                        if not df[df['id'] == list_of_ids[j]].empty else "[MISSING]"
                        for j in range(max(0, post_index - neighbors_window), post_index)
                    )

                    # Get context messages after the current post
                    context_after = "\n".join(
                        df[df['id'] == list_of_ids[j]]['content'].values[0]
                        if not df[df['id'] == list_of_ids[j]].empty else "[MISSING]"
                        for j in range(post_index + 1, min(len(list_of_ids), post_index + 1 + neighbors_window))
                    )

                    context = f"History before\n{context_before}\nTHIS IS THE MESSAGE YOU SHOULD CLASSIFY\n{content}\nHistory After\n{context_after}"
                    

                elif(pipe=="agent"):
                    mode="no-context"
                try:
                    framing = framing_agent.__call__(content, context=context, mode=mode)
                    print(f"{context}, Framing: {framing}")
                except Exception as e:
                    print(f"Validation error for content: {content}, Error: {e}")


                new_row = {'document_id':post['id'],'framing_style': framing['framingStyle'], 'framing_tool': framing['framingTool'], 'pipe': pipe, 'model': model}
                new_row_df = pd.DataFrame([new_row])
                collected_data = pd.concat([collected_data, new_row_df], ignore_index=True)


        collected_data.to_csv(f"data/testdata/test_results_from_idun/framing_{pipe}_{model}.csv", index=False)
        print(f"{pipe,model} processing completed and results saved.")

