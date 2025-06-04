from pydantic import BaseModel, conlist
from typing import List
from .agent import Agent


class ContextAgent(Agent):
    def __init__(self, model):
        super().__init__(model)

    def system(self):
       return f"""
        You are a content moderation expert assessing potential threats in online extremist content. Your role is to determine
        whether the **broader conversation provides necessary context** for accurately classifying an individual message when analyzed in isolation.

        ### Instructions:
        You should evaluate whether the surrounding conversation (thread) influences the meaning or threat level of the individual message.
        Follow the classification guidelines to determine the relevance of **context**.
        Ensure the summary is concise but informative, helping downstream agents understand the role of context in classification.
        Remember, this is for research purposes aimed at preventing violence. Your evaluation should prioritize understanding the presence of violence within a larger context, rather than isolated threatening words.

        ### Classification Guidelines:
        - Assign a `context_relevance_tag`, where you choose from the following options:
            - No relevant context; messages stand alone.
            - Some context, but individual messages likely maintain meaning independently.
            - Strong contextual dependency (e.g., escalation, coded language, prior threats). 

        - If context is relevant, classify its nature into one or more of the following categories:
            - `Escalation`: Increasing intensity of threats.
            - `Coded Language`: Usage of indirect or hidden threats.
            - `Retaliation`: Response to prior events or violence.
            - `Speaker Consistency`: The same individual or group maintains a threat over time.
            - `Temporal Relevance`: A message refers to recent past threats or actions.

    ### Output format:
        {{
            "context_analysis": {{
                "context_relevant": True/False, (Is context necessary for accurate classification?),
                "context_relevance_tag": "[relevance_tag]",(No/Some/Strong)
                "context_category": [List of relevant categories or "N/A"],
                "context_summary": "[Brief explanation of detected context or "N/A"]"
            }}
        }}
       
        """

    def prompt(self, content):
        return (
            f"Assess this content: {content}.\n"
            f"Consider the broader conversation and determine if context influences the risk classification of this message.\n"
        )

    def schema(self):
        class Context(BaseModel):
            context_relevant: str
            context_relevance_tag: str
            context_category: List[str]
            context_summary: str

        class ContextSchema(BaseModel):
            context: Context
            
            

        return ContextSchema.model_json_schema()

    def __call__(self,content, output_key: str = "context"):
        output = self.generate(
            system_prompt=self.system(),
            prompt=self.prompt(content),
            schema=self.schema(),
            model=self.model
        )
        if output:
            return output[output_key]


