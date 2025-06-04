from src.agents.legacy.call_to_action_agent import CallToActionAgent
from src.agents.framing_agent import FramingAgent
from src.agents.target_group_agent import TargetGroupAgent
from ..agents import ExampleAgent
from ..agents import EmotionAgent
from ..agents.legacy.otherness_agent import OthernessAgent
from ..agents.intent_agent import IntentAgent
from ..agents.classification_agent import MessageValidationAgent
from ..utils.rekey_dictionary import rekey_dict
from json import loads

import pandas as pd

model = "mistral-small"

data = pd.read_csv("data/grouped_data_from_stormfront/grouped_stormfront_data_2014_4.csv")

data_random_3 = data.sample(n=3)

collected_data = pd.DataFrame(columns=['document_id','num_posts_in_same_topic','topic_length','topic_violence_label','violence_label','intent_label','call_to_action','flagged_issues'])

otherness_agent = OthernessAgent(model)
framing_agent = FramingAgent(model)
intent_agent = IntentAgent(model)
validation_agent = MessageValidationAgent(model)
call_to_action_agent = CallToActionAgent(model)

for index,row in data_random_3.iterrows():

    content_with_ids = loads(row['content_user_list'])
    content = rekey_dict(content_with_ids)
    topic = row['stormfront_topic']
    num_posts = row['num_posts']
    topic_length = row['content_length']

    print("------------------------------")
    print("topic: \n", topic)
    print("------------------------------")
    print("number of posts: ",row['num_posts'])
    print("Size: ",row['content_length'])
    print("------------------------------")
    
    ### AVOID DOUBLE ANALYSIS IF ONLY ONE POST IN TOPIC ###
    topicWasAnalysed = False
    if(num_posts > 1):
         ### CHECKING FOR SIGNS OF 'OTHERNESS' ###
        otherness = otherness_agent.__call__(content)
        print(otherness)

        ### CHECKING FOR HIDDEN MEANINGS ###
        framing = framing_agent.__call__(content)
        print(framing)

        ### CHECKING FOR INTENT OF VIOLENCE ###
        intent = intent_agent.__call__(content, otherness['targetGroup'], framing)
        print(intent)

        ### CHECKING FOR CALL TO ACTION ###
        call_to_action = call_to_action_agent.__call__(content, otherness['targetGroup'], framing)
        print(call_to_action)

        validation = validation_agent.__call__(content, otherness_boolean = otherness['othernessBoolean'], target_group = otherness['targetGroup'], framing_style = framing['framingStyle'], framing_tool = framing['framingTool'], intent_of_violence=intent, call_to_action=call_to_action)
        print(validation)

        topicWasAnalysed = True

    
    print(" ------ ENTER THE THREAD ------")
    for post_docid in content_with_ids:
        print(" ------ NEW POST ------")
        print("\n" ,content_with_ids[post_docid])
        specific_post_otherness = otherness_agent.__call__(content_with_ids[post_docid]['content'])
        print(specific_post_otherness)

        specific_post_framing = framing_agent.__call__(content_with_ids[post_docid]['content'])
        print(specific_post_framing)

        specific_post_intent = intent_agent.__call__(content_with_ids[post_docid]['content'], specific_post_otherness['targetGroup'], specific_post_framing)
        print(specific_post_intent)

        specific_post_call_to_action = call_to_action_agent.__call__(content_with_ids[post_docid]['content'], otherness['targetGroup'], framing)
        print(specific_post_call_to_action)

        specific_post_validation = validation_agent.__call__(content_with_ids[post_docid]['content'], otherness_boolean = specific_post_otherness['othernessBoolean'], target_group = specific_post_otherness['targetGroup'], framing_style = specific_post_framing['framingStyle'], framing_tool = specific_post_framing['framingTool'], intent_of_violence=specific_post_intent, call_to_action=specific_post_call_to_action)
        print(specific_post_validation)

        if(topicWasAnalysed):
            new_row = {'document_id': post_docid, 'num_posts_in_same_topic': num_posts, 
               'topic_length': topic_length, 'topic_violence_label': validation['validated_label'], 
               'violence_label': specific_post_validation['validated_label'], 'intent_label': intent, 
               'call_to_action': call_to_action, 'flagged_issues': specific_post_validation['flagged_issues']}
        else:
            new_row = {'document_id': post_docid, 'num_posts_in_same_topic': num_posts, 
               'topic_length': topic_length, 'topic_violence_label': None, 
               'violence_label': specific_post_validation['validated_label'], 'intent_label': intent, 
               'call_to_action': call_to_action, 'flagged_issues': specific_post_validation['flagged_issues']}

        # Convert new_row to a DataFrame and concatenate with the existing DataFrame
        new_row_df = pd.DataFrame([new_row])
        collected_data = pd.concat([collected_data, new_row_df], ignore_index=True)

        
### COLLECTION OF DATA ###

collected_data.to_csv("data/collected_data.csv",index=False)


##### OLD CODE ######
    #Emotion analysis
    #emotion_agent = EmotionAgent(model)
    #emotions = emotion_agent.__call__(message)
    #print(emotions)

    #Target group analysis
    # target_group_agent = TargetGroupAgent(model)
    # target_group = target_group_agent.__call__(message)
    # print(target_group)