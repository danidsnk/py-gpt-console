import os
import sys

import keyboard
import openai
from openai.error import OpenAIError
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel

openai.api_key = os.getenv("OPENAI_API_KEY")


class ChatGpt:
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

    def __print_api_error(self, error: OpenAIError):
        print(f'OpenAIError: {error}')

    def chat_stream(self, prompt: str):
        self.__message_history.append({'role': 'user', 'content': prompt})
        try:
            res = ''
            for chunk in self.__gpt_stream():
                token = self.__get_token(chunk)
                res += token
                yield token

            self.__message_history.append({'role': 'assistant',
                                           'content': res})
        except OpenAIError as e:
            self.__print_api_error(e)

    def clear_history(self):
        self.__message_history = [self.__system_prompt]

    def set_system_prompt(self, system_prompt: str):
        self.__system_prompt['content'] = system_prompt
        self.clear_history()

    def raw_last_response(self) -> str:
        if len(self.__message_history) == 1:
            return ''
        return self.__message_history[-1]['content']


SYSTEM_PREFIX = '[ System ]: '
USER_PREFIX = '[ User ]: '
BOT_PREFIX = '[ Bot ]: '


def rich_chat(gpt: ChatGpt, prompt: str):
    with Live(refresh_per_second=12) as live:
        response = ''
        for token in gpt.chat_stream(prompt):
            response += token
            live.update(Panel(Markdown(response), title='GPT response'))


def multiline_input(prefix: str) -> str:
    sys.stdout.write(prefix)
    sys.stdout.flush()
    lines = ''
    for line in sys.stdin:
        lines += line
        if keyboard.is_pressed('ctrl'):
            break
    return lines


class ChatCommand:
    def __init__(self, gpt_console: ChatGpt):
        self.__gpt: ChatGpt = gpt_console

    def __call__(self, command: str) -> bool:
        match command:
            case '!exit':
                return False
            case '!clear':
                self.__gpt.clear_history()
            case '!system':
                sysprompt = multiline_input(SYSTEM_PREFIX)
                if sysprompt:
                    self.__gpt.set_system_prompt(sysprompt)
            case '!raw':
                print(BOT_PREFIX + self.__gpt.raw_last_response())
        return True


if __name__ == '__main__':
    gpt = ChatGpt()
    command = ChatCommand(gpt)
    while True:
        try:
            prompt = multiline_input(USER_PREFIX)
            if prompt.startswith('!'):
                if not command(prompt.strip()):
                    break
                continue
            if prompt.strip() == '':
                continue
            rich_chat(gpt, prompt)
        except Exception as e:
            print(f'Unexpected error: {e}')
