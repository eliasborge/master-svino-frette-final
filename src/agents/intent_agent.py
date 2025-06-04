from pydantic import BaseModel, conlist
from typing import List
from .agent import Agent


class IntentAgent(Agent):
    def __init__(
        self, model, 
    ):
        super().__init__(model)
       

    def system(self):
        return f"""
        You are a content moderation expert specializing in detection of violent and extremist language.
        Your role is to assess whether a message shows signs of **intent or call to action to cause harm or incite violence** toward a group.

        ### Instructions:
        You will analyze the message and determine if it shows signs of intent of violence or call to action rhetoric and give a string depending on the classification guidelines.
        Your specific task is to process a message and determine if the message endorses or expresses intent of aggression or violence.
        Your response must follow the exact output format. Do not add explanation or commentary.
        This task is part of a research pipeline aimed at preventing violence. Your analysis should be based on clear evidence of intent — not just tone, emotion, or threatening words taken in isolation.

        ### Classification Guidelines:
        **No intent of violence**
            - The message does not express hostility, aggression, or intent to cause harm towards any group.
        **Low intent of violence**
            - The message contains negative, hostile, or dehumanizing language directed at a group but does not imply a desire for harm or violence. This may include insults, derogatory remarks, or ideological opposition without direct calls for harm.
        **Moderate intent of violence**
            - The message suggests harm or justifies violence against a group, but does not directly incite it. This may include statements endorsing discrimination, celebrating harm, or implying that violence would be acceptable or inevitable.
        **High intent of violence**
            - The message explicitly encourages, calls for, or justifies violence against a group. This includes clear incitement, direct threats, or language that promotes violent action.

        **No call-to-action** 
            – The message does not encourage or suggest any action be taken by others.
        **Low call-to-action** 
            – The message subtly encourages action but lacks direct instructions or urgency.
        ** Moderate call-to-action**
            – The message contains a clear call for action but is non-violent or lacks forceful urgency.
        **High call-to-action**
            – The message explicitly calls for immediate or forceful action, potentially urging harmful or extreme measures.

            
        ### Output Format:
        {{
            "intent_of_violence": "No Intent of violence" / "Low Intent of violence" / "Moderate Intent of violence" / "High Intent of violence",
            "call_to_action": "No Call to Action" / "Low Call to Action" / "Moderate Call to Action" / "High Call to Action"

        }}
        """


    def prompt(self, content, framing, context, mode):

        if(mode=="no-context"):
                
            if (str(framing['framingStyle']).lower() == "implicit"):
                return f"""
                The message you are to analyze for intent of violence and call to action rhetoric is as follows: {content}.
                Analyze the message to determine if it shows signs of violent intention.
                Be aware the message may contain hidden meanings by the use of {framing['framingTool']}
                """
            else:
                return f"""
                The message you are to analyze for intent of violence and call to action rhetoric is as follows: {content}.
                Analyze the message to determine if it shows signs of violent intention.
                """
        elif(mode=="context"):
                
            if (str(framing['framingStyle']).lower() == "implicit"):
                return f"""
                You have been given a message that is a part of a broader conversation. This conversation has been analyzed by a context agent
                to provide you with insights into how relevant the surrounding messages are to the classification of this message.
                The context is as follows: {context}

                The message you are to analyze for intent of violence and call to action rhetoric is as follows: {content}.
                Analyze the message to determine if it shows signs of violent intention.
                Be aware the message may contain hidden meanings by the use of {framing['framingTool']}
                """
            else:
                return f"""
                You have been given a message that is a part of a broader conversation. This conversation has been analyzed by a context agent
                to provide you with insights into how relevant the surrounding messages are to the classification of this message.
                The context is as follows: {context}
                
                The message you are to analyze for intent of violence and call to action rhetoric is as follows: {content}.
                Analyze the message to determine if it shows signs of violent intention.
                """
            
        elif(mode=="neighbor"):
            if (str(framing['framingStyle']).lower() == "implicit"):
                return f"""
                You have been given a message that is a part of a broader conversation.
                The context is as follows: {context}

                The message you are to analyze for intent of violence and call to action rhetoric is as follows: {content}.
                Analyze the message to determine if it shows signs of violent intention.
                Be aware the message may contain hidden meanings by the use of {framing['framingTool']}
                """
            else:
                return f"""
                You have been given a message that is part of a series of neighboring messages. These neighboring messages provide additional context that may help in the classification of the message.
                The thread as follows: {context}
                
                The message you are to analyze for intent of violence is as follows: {content}.
                Analyze the message to determine if it shows signs of violent intention.
                """
    def schema(self):
        

        class Intent(BaseModel):
            intent_of_violence: str
            call_to_action: str

        class IntentSchema(BaseModel):
            intent: Intent

        return IntentSchema.model_json_schema()

    def __call__(self,content:str, framing:dict, context, mode:str, output_key: str = "intent"):
        output = self.generate(
            system_prompt=self.system(),
            prompt=self.prompt(content, framing, context, mode),
            schema=self.schema(),
            model=self.model
        )
        if output:
            return output[output_key]
