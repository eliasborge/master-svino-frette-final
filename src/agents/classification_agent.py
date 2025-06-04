from pydantic import BaseModel, conlist
from typing import List, Literal
from .agent import Agent


class ClassificationAgent(Agent):
    def __init__(self, model):
        super().__init__(model)

    def system(self):
        return f"""
        You are a Classification Agent. Your role is to verify and finalize the classification of social media posts, ensuring logical consistency and completeness across agent outputs. You must reason through your decisions before assigning a single label to each slot.

        **Classification Scope:**
        You will consider the following agent outputs:
        - **FramingAgent**: Identifies whether the content explicitly states its intentions, or if there are implicit meanings (e.g., sarcasm, irony).
        - **IntentAgent**: Assesses if there is direct violent intent or call to action rhetoric directed at someone.


        **Classification Rules:**
        - Content should **only** be classified as 'Violence risk' (**label 2**) if there is **clear intent to harm** either oneself, an individual or a group.
        - If the message contains **aggressive language, hate speech or hostility, but does not show significant implicit or explicit intent of violence, it should **not** be classified as 'Violence risk' (label 2). 
            Instead, consider **'Extremist but no violence risk' (label 1)**. 
        - If the classification is **uncertain**, flag the message for further review.

        **Error Handling & Flags:**
        1: If there are inconsistencies between the agent outputs, use **flag 0**.
        2: If the outputs are consistent, but you are unsure about your final decision -> classification is **uncertain**, use **flag 1**.
        3: If the content in your opinion does **not align** with the classification, use **flag 2**, and label it uncertain.
        4: If you are unable to process the content due to content filters, use **flag 3**.

        **Labeling:**
        - **Answer 0** → 'No violence risk'
        - **Answer 1** → 'Extremist but no violence risk'
        - **Answer 2** → 'Violence risk' (only if explicit intent is undeniably present)

        **Output Format:**
        {{
            "classification": {{
                "label": "0/1/2",
                "flagged_issues": [0/1/2/3/4]
            }}
        }}
        """

    def prompt(self, content, framing_style, framing_tool, intent_of_violence, call_to_action, context,mode):
        
        if mode == "no-context":
            return (
                f"Assess the classification of this content: {content}.\n"
                f"Given the analysis below, reason thoroughly before assigning a classification:\n"
                f"- Framing: {framing_style} using {framing_tool}\n"
                f"- Intent of Violence: {intent_of_violence}\n"
                f"- Call to Action: {call_to_action}\n\n"
              
            )
        elif(mode=="context"):
            return (
                f"You have been given a message that is part of a broader conversation. This conversation has been analyzed by a context agent "
                f"to provide insights into the classification of this message. The context analysis is as follows: {context}\n\n"

                f"Assess the classification of this content: {content}.\n"
                f"Given the analysis below, reason thoroughly before assigning a classification:\n"
                f"- Framing: {framing_style} using {framing_tool}\n"
                f"- Intent of Violence: {intent_of_violence}\n"
                f"- Call to Action: {call_to_action}\n\n"
                
            )
        
        elif(mode=="neighbor"):
             return (
                f"You have been given a message that is part of a series of neighboring messages. These neighboring messages provide additional context that may help in the classification of the message."
                f"The thread as follows: {context}"

                f"Assess the classification of this content: {content}.\n"
                f"Given the analysis below, reason thoroughly before assigning a classification:\n"
                f"- Framing: {framing_style} using {framing_tool}\n"
                f"- Intent of Violence: {intent_of_violence}\n"
                f"- Call to Action: {call_to_action}\n\n"
               
             )
    def schema(self):
        class Classification(BaseModel):
            label: Literal[0, 1, 2]
            flagged_issues: List[int]

        class ClassificationSchema(BaseModel):
            classification: Classification

        return ClassificationSchema.model_json_schema()

    def __call__(self, content, framing_style, framing_tool, intent_of_violence, call_to_action, context,mode:str, output_key: str = "classification"):
        output = self.generate(
            system_prompt=self.system(),
            prompt=self.prompt(content, framing_style, framing_tool, intent_of_violence, call_to_action, context,mode),
            schema=self.schema(),
            model=self.model
        )
        if output:
            return output[output_key]
