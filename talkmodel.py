from openai import OpenAI


class talkmodel:
    client = None

    def __init__(self) -> None:
        self.client = OpenAI(
            base_url='http://localhost:11434/v1/',
            # required but ignored
            api_key='ollama',
        )

    def chat(self, message):
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    'role': 'system',
                    'content': 'You are a helpful assistant. please reply in Chinese.',
                },
                {
                    'role': 'user',
                    'content': message,
                }
            ],
            model='llama2',
        )

        return chat_completion.choices[0].message.content
    
if __name__ == "__main__":
    talk = talkmodel()
    print(talk.chat("hello"))