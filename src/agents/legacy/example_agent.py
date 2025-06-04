from pydantic import BaseModel, conlist
from typing import List
from ..agent import Agent


class ExampleAgent(Agent):
    def __init__(
        self, model, 
    ):
        super().__init__(model)
       

    def system(self):
        return f"You are an example agent and your task is to verify that the system works"

    def prompt(self):
        return f"Write 1 if the system works and 0 if something went wrong"

    def schema(self):
        class VerificationMessage(BaseModel):
            message: str

        class VerificationSchema(BaseModel):
            verification: VerificationMessage

        return VerificationSchema.model_json_schema()

    def __call__(self, output_key: str = "verification"):
        output = self.generate(
            system_prompt=self.system(),
            prompt=self.prompt(),
            schema=self.schema(),
            model=self.model
        )
        if output:
            return output[output_key]

