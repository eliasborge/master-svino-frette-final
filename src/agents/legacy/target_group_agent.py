from pydantic import BaseModel, Field
from typing import List
from ..agent import Agent


class TargetGroupAgent(Agent):
    def __init__(
        self, model
    ):
        super().__init__(model)
       
       

    def system(self):
        return f"""You are an AI agent tasked with identifying whether a given message targets a specific group or individual. 
        This can include any minority, demographic, or other identifiable group. 
        If the target group is not clear, respond with "unclear". 
        If the message is offensive, respond with "offensive". 
        Additionally, provide your certainty level on a scale from 0 to 1, where 0 is unsure and 1 is very sure.
        For example, if the message is "I hate white people", the target group is "white people" and the certainty level is 0.9.
        If more than one demographic is mentioned, focus on the one that is mentioned negatively.
        """

    def prompt(self, content: str):
        return f"Analyse the given message and determine if there is a specific target group in it. Reply with [target group], unclear or offensive. The message is: {content}"

    def schema(self):
        class TargetGroupSchema(BaseModel):
            target_group: str
            certainty: float
            
        return TargetGroupSchema.model_json_schema()

    def __call__(self,content, output_key: str = "output"):
        output = self.generate(
            system_prompt=self.system(),
            prompt=self.prompt(content),
            schema=self.schema(),
            model=self.model
        )
        print(output)
        if output:
            return output

   