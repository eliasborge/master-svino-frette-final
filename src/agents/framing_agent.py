from pydantic import BaseModel, conlist
from typing import List
from .agent import Agent


class FramingAgent(Agent):
    def __init__(
        self, model, 
    ):
        super().__init__(model)
       

    def system(self):
        return f"""
        You are a language analysis expert in a moderation pipeline tasked with evaluating whether messages contain **explicit or implicit meaning**.
        Your goal is to identify whether a message expresses its meaning clearly, or if it relies on **hidden cues** such as **sarcasm, irony, metaphor, or coded language**. 
        These are considered *implicit framing styles*.

        ### Instructions:
        This analysis is performed to ensure accurate understanding of the message content, and it is therefore important that you thoroughly analyze the message and not just provide a random answer. 
        Your specific task is to process a message and determine if it contains any hidden meanings that are not explicitly stated.
        You will analyze the message and determine if it shows signs of hidden meanings and provide a true or false value along with the identified hidden meaning if any.
        Do not guess randomly. If a message is ambiguous or lacks clarity, indicate that explicitly.
        If the message is deemed as Implicit, there should always be a related FramingTool. If the message is deemed as Explicit, the FramingTool should be empty.
        Always answer something. Either you give the answer or you indicate that you are unsure.
        Remember, this is for research purposes aimed at preventing violence. Your analysis should focus on the presence of hidden meanings, rather than just the use of threatening words.

        ### Classification Guidelines:
        1. **Framing Style** (`framingStyle`):
        - `"Explicit"` → The meaning is directly stated and clearly understood.
        - `"Implicit"` → The message relies on hidden meaning (e.g., sarcasm, metaphor, irony).
        - If the message is too ambiguous to assess reliably, still return a valid output and indicate ambiguity in the next field.

        2. **Framing Tool** (`framingTool`):
        - If `framingStyle` is `"Explicit"` → Leave this field **empty**.
        - If `framingStyle` is `"Implicit"` → Specify the tool(s) used (e.g., `"Sarcasm"`, `"Irony"`, `"Metaphor"`, `"Coded Language"`).
        - If the message is too ambiguous, return `"Ambiguous"` in this field.

        
        ### Output Format:
        {{
        "framing": 
            {{
            "framingStyle": "Explicit" or "Implicit",
            "framingTool": "<related tool or 'Ambiguous' or ''>"
            }}
        }}
        """
        

    def prompt(self,content, context, mode):
        if(mode=="no-context"):
                    
            return f"""
            The message you are to analyze for hidden meanings is as follows: {content}.

            """

        elif(mode=="context"):
            
            return f"""
            You have been given a message that is a part of a broader conversation. This conversation has been analyzed by a context agent
            to provide you with insights into how relevant the surrounding messages are to the classification of this message.
            The context is as follows: {context}

            The message you are to analyze for hidden meanings is as follows: {content}.

            """
        
        elif(mode=="neighbor"):
            
            return f"""
            You have been given a message that is part of a series of neighboring messages. These neighboring messages provide additional context that may help in the classification of the message.
            The thread as follows: {context}

            The message you are to analyze for hidden meanings is as follows: {content}.

            """

    def schema(self):
        class FramingAnalysis(BaseModel):
            framingStyle: str
            framingTool:str

        class FramingSchema(BaseModel):
            framing: FramingAnalysis

        return FramingSchema.model_json_schema()

    def __call__(self, content, context, mode:str, output_key: str = "framing"):
        output = self.generate(
            system_prompt=self.system(),
            prompt=self.prompt(content, context, mode),
            schema=self.schema(),
            model=self.model
        )
        if output:
            return output[output_key]

