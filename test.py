from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

import lang_interface

lang_interface.DEBUG = True
load_dotenv()
llm = OpenAI()


def llm_function(messages) -> str:
    resp = llm.chat.completions.create(messages=messages, model='gpt-4o')
    return resp.choices[0].message.content


class Friend(BaseModel):
    full_name: str
    phone: str
    height: str
    weight: str
    address: str
    age: int
    is_best: bool = False


class MyInterface:
    """API for retrieving into about user's friends"""
    all_friends = {
        "John": {
            "full_name": "John Doe",
            "phone": "123-456-7890",
            "height": "180 cm",
            "weight": "75 kg",
            "address": "123 Maple Street, Springfield, IL 62704",
            "age": 30,
            "is_best": True,
        },
        "Jane": {
            "full_name": "Jane Smith",
            "phone": "987-654-3210",
            "height": "165 cm",
            "weight": "60 kg",
            "address": "456 Oak Avenue, Springfield, IL 62704",
            "age": 28,
            "is_best": True,
        },
        "Alice": {
            "full_name": "Alice Johnson",
            "phone": "555-123-4567",
            "height": "170 cm",
            "weight": "68 kg",
            "address": "789 Pine Road, Springfield, IL 62704",
            "age": 32,
            "is_best": False,
        },
        "Bob": {
            "full_name": "Bob Brown",
            "phone": "444-555-6666",
            "height": "175 cm",
            "weight": "80 kg",
            "address": "321 Birch Lane, Springfield, IL 62704",
            "age": 35,
            "is_best": False,
        }
    }

    def do_get_friend_list(
        self, best_only: bool = False,
        name_starts_with: str = None,
        limit: int = 10
    ) -> list[str]:
        """
        Return list of all user's friends'
        Caution: maximum limit is 20
        """
        if not best_only:
            friends = list(self.all_friends.keys())
        else:
            friends = [
                name for name, data in self.all_friends.items()
                if data['is_best']
            ]
        if name_starts_with:
            friends = [
                          f for f in friends if f.startswith(name_starts_with)
                      ][:limit]
        else:
            friends = friends[:limit]
        return friends

    def do_get_friend_info(self, friend_name: str) -> Friend:
        """Return list of all user's friends'"""
        return Friend(**self.all_friends[friend_name])

    def do_make_friend_best(self, friend_name: str) -> str:
        """Sets a friend as a best friend"""
        self.all_friends[friend_name]['is_best'] = True
        return "Now it is your best friend"

    def do_set_friend_property(
        self, friend_name: str, property: str, value
    ) -> str:
        """Sets a friend as a best friend"""
        self.all_friends[friend_name][property] = value
        return f"Property {property} is set to {value} for {friend_name}"

    def do_delete_from_friends(self, friend_name: str) -> str:
        if friend_name in self.all_friends:
            return f"You don't have friend with name {friend_name}"
        del self.all_friends[friend_name]
        return f"Friend {friend_name} deleted"


assistant = lang_interface.Assistant(
    MyInterface(),
    llm=llm_function  # noqa
)


def test_chat():
    while True:
        try:
            q = input('\033[1;36m> ')
            print('\033[0m', end='')
            answer = assistant(q)
            print(f'\033[0massistant: {answer}')
        except KeyboardInterrupt:
            print('\033[0;32mbuy')


if __name__ == '__main__':
    test_chat()
