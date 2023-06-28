import os

import openai
from openai.error import OpenAIError
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel

openai.api_key = os.getenv("OPENAI_API_KEY")


class RichGptConsole:
    def __init__(self, system_prompt: str = 'You are helpful assistant'):
        self.__system_prompt = {'role': 'system', 'content': system_prompt}
        self.__message_history = [self.__system_prompt]

    def __gpt_stream(self):
        return openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=self.__message_history,
            # max_tokens=100,
            stream=True,
        )

    def __get_token(self, response):
        if 'delta' in response.choices[0]:
            delta = response.choices[0].delta
            if 'content' in delta:
                return delta.content
        return ''

    def __response_processing(self, prompt: str, live_rich: Live):
        self.__message_history.append({'role': 'user', 'content': prompt})
        res = ''
        for chunk in self.__gpt_stream():
            res += self.__get_token(chunk)
            live_rich.update(Panel(Markdown(res), title='GPT response'))

        self.__message_history.append({'role': 'assistant', 'content': res})

    def __print_api_error(self, error: OpenAIError):
        print(f'Error: {error}')

    def rich_chat(self, prompt: str):
        try:
            with Live(refresh_per_second=12) as live:
                self.__response_processing(prompt, live)
        except OpenAIError as e:
            self.__print_api_error(e)

    def clear_history(self):
        self.__message_history = [self.__system_prompt]

    def set_system_prompt(self, system_prompt: str):
        self.__system_prompt['content'] = system_prompt
        self.clear_history()

    def raw_last_response(self) -> str:
        return self.__message_history[-1]['content']


class _Commands:
    def __init__(self, gpt_console: RichGptConsole):
        self.__gpt: RichGptConsole = gpt_console

    def __call__(self, command: str) -> bool:
        match command:
            case '!exit':
                return False
            case '!clear':
                self.__gpt.clear_history()
            case '!system':
                sysprompt = input('[ System ]: ')
                if '!multi' == sysprompt:
                    sysprompt = self.__multiline_input()
                if sysprompt:
                    self.__gpt.set_system_prompt(sysprompt)
            case '!raw':
                print(self.__gpt.raw_last_response())
            case '!multi':
                multiline = self.__multiline_input()
                if multiline:
                    self.__gpt.rich_chat(multiline)

        return True

    def __multiline_input(self) -> str:
        lines = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            lines.append(line)
        return '\n'.join(lines)


if __name__ == '__main__':
    gpt = RichGptConsole()
    command = _Commands(gpt)
    while True:
        prompt = input('[ User ]: ')
        if prompt.startswith('!'):
            if not command(prompt):
                break
            continue
        elif prompt == '':
            continue
        gpt.rich_chat(prompt)
