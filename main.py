import os

import openai
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel

openai.api_key = os.getenv("OPENAI_API_KEY")


class RichGptConsole:
    def __init__(self, system_prompt='You are helpful assistant'):
        self.__system_prompt = {'role': 'system', 'content': system_prompt}
        self.__message_history = [self.__system_prompt]

    def __gpt_stream(self):
        return openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=self.__message_history,
            # max_tokens=100,
            stream=True,
        )

    def __get_part(self, response):
        if 'delta' in response.choices[0]:
            delta = response.choices[0].delta
            if 'content' in delta:
                return delta.content
        return ''

    def __response_processing(self, prompt, live_rich):
        self.__message_history.append({'role': 'user', 'content': prompt})
        res = ''
        for response in self.__gpt_stream():
            res += self.__get_part(response)
            live_rich.update(Panel.fit(Markdown(res), title='GPT response'))

        self.__message_history.append({'role': 'assistant', 'content': res})

    def chat(self, prompt):
        with Live(refresh_per_second=12) as live:
            self.__response_processing(prompt, live)

    def clear_history(self):
        self.__message_history = [self.__system_prompt]

    def set_system_prompt(self, system_prompt):
        self.__system_prompt['content'] = system_prompt
        self.clear_history()


if __name__ == '__main__':
    gpt = RichGptConsole()
    while True:
        prompt = input('[ User ]: ')
        if prompt == '!exit':
            break
        elif prompt == '!clear':
            gpt.clear_history()
            continue
        elif prompt.startswith('!system'):
            gpt.set_system_prompt(prompt[8:])
            continue
        elif prompt == '':
            continue
        gpt.chat(prompt)
