from src.model_config import AVAILABLE_MODELS
from .agents.batch_agent import BatchAgent
from datetime import datetime
import pandas as pd

import time


### Configuration
model = AVAILABLE_MODELS[0]  # mistral
BATCH_SIZE = 5

### Logging Setup
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
start_time = time.time()

### Data Loading
grouped_df = pd.read_csv("data/testdata/grouped_processed_VideoCommentsThreatCorpus.csv")

### Agent Initialization
batch_agent = BatchAgent(model)

### Output Storage Initialization
collected_data = pd.DataFrame(columns=['video_num', 'document_id', 'violence_label', 'flagged_issues'])
efficiency_data = pd.DataFrame(columns=[
    'row', 'row_duration_sec'
])

### Batch Processing
print("Starting batch processing...")
for index, row in grouped_df.iterrows():
    print(f"Processing row {index + 1} of {len(grouped_df)}...")

    row_start_time = time.time()

    # Extract content and IDs
    content_list = row['content'].split("###---###")
    list_of_ids = row['id'].split(", ")
    results = []

    # Process in batches
    for i in range(0, len(content_list), BATCH_SIZE):
        chunk = content_list[i:i + BATCH_SIZE]
        chunk_text = "\nNew post:\n".join(chunk)
        result = batch_agent.__call__(chunk_text)

        if not isinstance(result, list):
            result = [result]
        results.extend(result) 

    # Collect results
    for i, flag in enumerate(results):
        new_row = {
            'video_num': list_of_ids[i].split('_')[0],
            'document_id': list_of_ids[i],
            'violence_label': flag.get('violent_label', -1),
            'flagged_issues': flag.get('flagged_issues', "None")
        }
        collected_data = pd.concat([collected_data, pd.DataFrame([new_row])], ignore_index=True)


    row_duration = time.time() - row_start_time

    efficiency_row = {
        'row': index + 1,
        'row_duration_sec': row_duration
    }
    efficiency_data = pd.concat([efficiency_data, pd.DataFrame([efficiency_row])], ignore_index=True)

### Save Results
collected_data.to_csv(f"data/testdata/test_results_from_idun/batch/batch_{model}_{timestamp}.csv", index=False)
efficiency_data.to_csv(f"data/testdata/test_results_from_idun/batch/batch_efficiency_{model}_{timestamp}.csv", index=False)
print("Batch processing completed and results saved.")
