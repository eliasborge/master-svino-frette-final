from pydantic import BaseModel, Field
from typing import List
from ..agent import Agent


class EmotionAgent(Agent):
    def __init__(
        self, model
    ):
        super().__init__(model)
       
       

    def system(self):
        return f"""You are an AI agent that is tasked with classifying emotions and their intensities based on a given message that you are given.
        For each message you are to grade it on a scale from 0.0 to 1.0 for all of the given emotions: anger, disgust, fear, envy, desire.
        """

    def prompt(self, content: str):
        return f"Analyse the given message and classify the emotions present in the message and their intensities. The message is: {content}"

    def schema(self):
        class EmotionSchema(BaseModel):
            anger: float = Field(..., ge=0.0, le=1.0)
            disgust: float = Field(..., ge=0.0, le=1.0)
            fear: float = Field(..., ge=0.0, le=1.0)
            envy: float = Field(..., ge=0.0, le=1.0)
            desire: float = Field(..., ge=0.0, le=1.0)
            
        return EmotionSchema.model_json_schema()

    def __call__(self,content, output_key: str = "output"):
        output = self.generate(
            system_prompt=self.system(),
            prompt=self.prompt(content),
            schema=self.schema(),
            model=self.model,
            num_ctx=200,
            temperature=0.0,
        )
        if output:
            return output

