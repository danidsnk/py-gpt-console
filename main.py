import sys
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

message_history = [
    # {'role':'system', 'content':''},
]


def ai_request(text: str):
    sys.stdout.write('Bot: ')
    assistant_response = {}
    user_request = {'role': 'user', 'content': text}
    message_history.append(user_request)
    res = ''
    for response in openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=message_history,
        # max_tokens=100,
        stream=True,
    ):
        if 'delta' in response.choices[0]:
            delta = response.choices[0].delta
            if 'role' in delta:
                assistant_response['role'] = delta.role
            if 'content' in delta:
                sys.stdout.write(delta.content)
                sys.stdout.flush()
                res += delta.content
    sys.stdout.write('\n')

    assistant_response['content'] = res
    message_history.append(assistant_response)


if __name__ == '__main__':
    print('Bot started!')
    while True:
        text = input('User: ')
        if text == 'exit':
            break
        elif text == '':
            continue
        ai_request(text)

    print(message_history)
