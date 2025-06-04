from pydantic import BaseModel, conlist
from typing import List
from ..agent import Agent


class OthernessAgent(Agent):
    def __init__(
        self, model, 
    ):
        super().__init__(model)
       

    def system(self):
        return f""" 
        You are a content moderation expert specializing in the detection of **otherness**. 
        Otherness is the concept of an "us-versus-them" mentality, where a group is framed as different, separate, or inferior.
        This framing can involve minorities, demographics, social groups, political affiliations, or any identifiable community.  

        ### Instructions:
        - Otherness is only True if there is negative framing of another group.  
        - If a message is neutral or positive towards a group without attacking another, otherness should be False.  
        - If the message contains positive sentiment about an in-group (e.g., "I love my own group"), otherness should be False unless it simultaneously degrades another group.
        - Remember, this is for research purposes aimed at preventing violence. Your analysis should focus on the presence of otherness, rather than just the use of threatening words.
        - If otherness is present, you must identify the target group. The response must always include a True or False value for otherness and the target group if otherness is True.
        - You cannot return an empty target group if otherness is True, as this is an inconsistency.

        ### Output Format:
        {{
            "otherness": {{
                "othernessBoolean": "True/False",
                "targetGroup": "group_name"
            }}
        }}
        """

    def prompt(self, content, context, mode):

        if(mode=="no-context"):
            
            return f"""
            The message you are to analyze for otherness is as follows: {content}.
            Give a thorough analysis on the message and determine if it shows signs of otherness.

            Provide a True or False value to the following statement: "The message shows signs of otherness" and identify the target group if any. Remember that otherness is true only if another group is framed negatively.
            If a group is mentioned neutrally or positively, otherness should be False. If a message expresses positive sentiment toward an in-group without attacking another group, otherness should be False.
            If you find the content too offensive, you will have to provide a response that indicates that the content is too offensive to analyze.
            Always answer something. If you find otherness to be true, there has to be an associated target group. i.e. you cannot answer otherness = true and target group = "", as this is an inconsistency
            """
        elif(mode=="context"):

            return f"""
            You have been given a message that is a part of a broader conversation. This conversation has been analyzed by a context agent
            to provide you with insights into how relevant the surrounding messages are to the classification of this message.
            The context is as follows: {context}
            
            The message you are to analyze for otherness is as follows: {content}.
            Give a thorough analysis on the message and determine if it shows signs of otherness.

            Provide a True or False value to the following statement: "The message shows signs of otherness" and identify the target group if any. Remember that otherness is true only if another group is framed negatively.
            If a group is mentioned neutrally or positively, otherness should be False. If a message expresses positive sentiment toward an in-group without attacking another group, otherness should be False.
            If you find the content too offensive, you will have to provide a response that indicates that the content is too offensive to analyze.
            Always answer something. If you find otherness to be true, there has to be an associated target group. i.e. you cannot answer otherness = true and target group = "", as this is an inconsistency
            """
        
        elif(mode=="neighbor"):
            return f"""
            You have been given a message that is part of a series of neighboring messages. These neighboring messages provide additional context that may help in the classification of the message.
            The thread as follows: {context}
            
            The message you are to is marked with "THIS IS THE MESSAGE YOU SHOULD CLASSIFY". The remaining messages are simply for context.
            Give a thorough analysis on the message and determine if it shows signs of otherness. The message is: {content}

            Provide a True or False value to the following statement: "The message shows signs of otherness" and identify the target group if any. Remember that otherness is true only if another group is framed negatively.
            If a group is mentioned neutrally or positively, otherness should be False. If a message expresses positive sentiment toward an in-group without attacking another group, otherness should be False.
            If you find the content too offensive, you will have to provide a response that indicates that the content is too offensive to analyze.
            Always answer something. If you find otherness to be true, there has to be an associated target group. i.e. you cannot answer otherness = true and target group = "", as this is an inconsistency
            """
            

    def schema(self):
        class OthernessAnalysis(BaseModel):
            othernessBoolean: str
            targetGroup: str

        class OthernessSchema(BaseModel):
            otherness: OthernessAnalysis

        return OthernessSchema.model_json_schema()

    def __call__(self,content:str, context, mode:str, output_key: str = "otherness"):
        output = self.generate(
            system_prompt=self.system(),
            prompt=self.prompt(content, context,mode),
            schema=self.schema(),
            model=self.model
        )
        if output:
            return output[output_key]

