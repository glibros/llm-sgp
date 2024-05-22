import os
from openai import OpenAI
from .gp import Library

class LLM(Library):
    """This class is used to store the context of a chat session with the llm assistant. 
    This implementation is based on the OpenAI API, but can be easily adapted to use other
    llm APIs."""

    def __init__(self, dimensions, regressions, seed=42, toolbox=None, points=None):
        """Initializes the chat context.

        :param dimensions: The number of dimensions to the SR problem.
        :param regressions: The regressions results obtained previously (priors)
        :param seed: The seed to use for the llm to "reproduce" experiment
        """
        Library.__init__(self, toolbox, points)
        # defaults to getting the key using os.environ.get("OPENAI_API_KEY")
        # self.client = OpenAI()
        self.client = OpenAI(
            base_url="http://localhost:8000/v1", # "http://<Your api-server IP>:port"
            api_key = "sk-no-key-required",
        )
        # Get grammars from grammar file
        with open(os.path.join(os.path.dirname(__file__), "../schema.gbnf"), "r") as f:
            self.grammars = f.read()
        self.dimensions = dimensions
        self.seed = seed
        self.regressions = regressions # For priors
        self.individuals = ""
        self.messages = [
                {"role": "system", "content": "You are a data scientist assistant skilled in "
                "optimizing Genetic Programming algorithms for a "
                "Symbolic Regression problem with: " + str(self.dimensions) + " input(s) and 1 output. ("
                + str(self.dimensions) + "-dimensional, one-based x indexing e.g. x1, x2, x3, ...) "},
        ]

    def reset(self):
        """Resets the chat context."""
        self.messages = [
                {"role": "system", "content": "You are a data scientist assistant skilled in "
                "optimizing Genetic Programming algorithms for a "
                "Symbolic Regression problem with: " + str(self.dimensions) + " input(s) and 1 output. ("
                + str(self.dimensions) + "-dimensional, one-based x indexing e.g. x1, x2, x3, ...) "},
        ]

    def initialAnalysis(self, user_message, assistant_message):
        """Adds initial user analysis to the chat context. This is used to set the tone of the 
        conversation, and give the assistant some context about the user's problem.

        :param user_message: The user's initial message.
        :param assistant_message: The assistant's initial message.
        """
        self.messages.append({"role": "user", "content": user_message})
        self.messages.append(
            {"role": "assistant", "content": assistant_message})
        return self.messages
    
    def complete(self, message, temperature=0.2, top_p=1.0):
        """ Ask llama to complete the message.

        :param message: The question to ask the assistant.
        """
        response = self.client.completions.create(
            model="llm_models",
            max_tokens=4096,
            temperature=temperature, # 0.2 is the default 
            top_p=top_p, # By default this is set to 1.0 in fastchat. Use this setting for black-box initialization
            prompt=("Below is an instruction that describes a task. " 
                    "Write a response that appropriately completes the request.\n\n" 
                    "### Instruction:\n")
                    + message + 
                    "\n\n" + "### Response: Let's think step by step. "
        )
        return response.choices[0].text

    def addQuestion(self, message):
        """Adds a question to the chat context.

        :param message: The question to ask the assistant.
        """
        self.messages.append({"role": "user", "content": message})
        response = self.client.chat.completions.create(
            model="gpt-4-1106-preview", # With llama2 it doesn't matter
            response_format={"type": "json_object"}, # Also doesn't matter
            extra_body={
                "grammar": self.grammars, 
                "n_predict": 2048, 
                "repeat_penalty": 2.0, 
                "repeat_last_n": 512, 
                "seed": self.seed},
            seed=self.seed,  # Because that's the meaning of life
            messages=self.messages,
            max_tokens=4096
        )
        self.messages.append(
            {"role": "assistant", "content": response.choices[0].message.content})
        return response.choices[0].message

    def addImages(self, images, message):
        local_content = [{"type": "text", "text": message}]
        for image in images:
            local_content.append({"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{image}"}})
        self.messages.append({"role": "user", "content": local_content})
        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            # response_format={"type": "json_object"},
            messages=self.messages,
            max_tokens=200
        )
        self.messages.append(
            {"role": "assistant", "content": response.choices[0].message.content})
        return response.choices[0].message

__all__ = ['LLM']