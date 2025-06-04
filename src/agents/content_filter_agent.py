from pydantic import BaseModel
from .agent import Agent

class ContentFilterAgent(Agent):
    def __init__(self, model):
        super().__init__(model)

    def system(self):
        return f"""
        You are an AI assistant designed to converse and interact with users.
        When a user sends a message, analyze it and respond to the request within the best of your abilities

        """

    def prompt(self, content):
        return content

    def schema(self):
      

        class ReplySchema(BaseModel):
            reply: str

        return ReplySchema.model_json_schema()


    def __call__(self, content, output_key: str = "reply"):
        output = self.generate(
            system_prompt=self.system(),
            prompt=self.prompt(content),
            schema=self.schema(),
            model=self.model
        )
        if output:
            return output[output_key]
            
