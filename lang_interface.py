import ast
import inspect
import re
from typing import Callable, Any

from tenacity import retry, stop_after_attempt

try:
    from pydantic import BaseModel

    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False

DEBUG = False

CHAT_PROMPT = """
You are an AI assistant connecting a user and a following python programming interface:
## Description:
{description}
## Operations:
{interface}

Your goal is to convert user's request into a python function call with parameters.
If user's request is clear respond in the following format:
<CALL>function(**parameters)</CALL>
If it's not clear what function to call and what parameters to pass, respond with clarification message:
<CLARIFY>question</CLARIFY>

Also, if one interface call is not enough to retrieve needed info, anyway provide the first required call.

User might ask questions, not related to provided interface, so act as a simple AI assistant, answer:
<ANOTHER>answer</ANOTHER>

"""  # noqa
FUNCTION_PROMPT = """
You are an AI assistant connecting a user and a following programming interface:

## Description:
{description}
## Operations:
{interface}

Your goal is to convert user's request into a python function call with parameters.
You will receive user's question, the last called operation and the output from that operations in the following format:
<QUESTION>question</QUESTION>
<OPERATION>operation</OPERATION>
<OUTPUT>output</OUTPUT>

You have three options:
1. Respond with a given output, if output from operation seem to satisfy user's request, but reformat it to make read-friendly and get cut not relevant data.
2. Call another operation, or same operation with different parameters
3. Respond with clarification question to a user, if output from operation is not the answer to his question, but it's not clear what operation needs to be called instead.

Depending on the option respond in 3 different ways:
1. <OUTPUT> output here </OUTPUT>
2. <CALL>function(**params)</CALL>
3. <CLARIFY>question</CLARIFY>

"""  # noqa
FUNCTION_PROMPT_USER = """
<QUESTION>question</QUESTION>
<OPERATION>{operation}</OPERATION>
<OUTPUT>{output}</OUTPUT>
"""  # noqa


class ProcessException(Exception):
    pass


class Assistant:
    """LLM Assistant interface for python class"""

    def __init__(
        self,
        handler,
        llm: Any | Callable,
        *,
        model: str = None,
        max_context_chars: int = 50_000,
        methods_prefix: str = '',
        max_chain_calls: int = 7,
        sorry_message: str = "Sorry, cannot process your request",
    ):
        """
        Init LLM Assistant interface for python class.
        :param handler: a class that implements a programing interface.
        :param llm: Can be either OpenAI client or any callable:
                def call(messages: list[dict]) -> str
        :param model: openai model name (only provide if llm is OpenAI client)
        :param max_context_chars: maximum number of chars allowed in a context.
        :param methods_prefix: prefix string to indicate public interface
            methods, if not specified, all public methods will be used.
        :param max_chain_calls: maximum number of LLM calls per user request.
        :param sorry_message: message to show when request
            couldn't be processed.
        """
        self.handler = handler
        self.llm = _get_llm_wrapper(llm, model)
        self.max_context = max_context_chars
        self._spec = _parse_interface(handler, methods_prefix)
        self._interface_md = _get_interface_md(self._spec)
        self._max_chain_calls = max_chain_calls
        self._sorry_message = sorry_message

        _debug('Interface:\n' + self._interface_md)

        description = inspect.getdoc(self.handler) or "Interface"
        self.chat_prompt = CHAT_PROMPT.format(
            interface=self._interface_md,
            description=description
        )
        self.function_prompt = FUNCTION_PROMPT.format(
            interface=self._interface_md,
            description=description
        )
        self._messages = [{'role': 'system', 'content': self.chat_prompt}]

    def __call__(self, question: str) -> str:
        try:
            return self.handle(question)
        except ProcessException as e:
            _debug(f"Failed to process user request {e}")
            return self._sorry_message

    def handle(self, question: str) -> str:
        self._messages.append({'role': 'user', 'content': question})
        self._trim_messages()
        final_result = None
        response = self._llm(messages=self._messages)
        action, content = _parse_response(response)

        if action in ['clarify', 'another']:
            final_result = content

        num_calls = 0
        while final_result is None and num_calls < self._max_chain_calls:
            output = self.call_method(content)
            action, content = self._process_output(action, output)
            if action != 'call':
                final_result = content
        if not final_result:
            final_result = self._sorry_message

        self._messages.append({'role': 'assistant', 'content': final_result})
        return final_result

    def clen_history(self):
        self._messages = self._messages[:1]

    def _trim_messages(self):
        char_used = 0
        new_messages = []
        for message in self._messages[:0:-1]:
            char_used += len(message['content'])
            if char_used < self.max_context:
                new_messages.append(message)
        new_messages = [self._messages[0]] + new_messages[::-1]
        self._messages = new_messages

    @retry(stop=stop_after_attempt(3))
    def _process_output(self, operation, output):
        messages = [
            {'role': 'system', 'content': self.function_prompt},
            *self._messages[1:],
            {'role': 'user', 'content': FUNCTION_PROMPT_USER.format(
                operation=operation,
                output=output,
                question=self._messages[-1]['content']
            )}
        ]
        resp = self._llm(messages=messages)
        return _parse_response(resp)

    def call_method(self, method_call):
        handler = self.handler

        # the only expression we allow is calling list(function())
        if re.match("list\\(.*\\)", method_call):
            method_call = f"list(handler.{method_call})"
        else:
            method_call = f'handler.{method_call}'
        try:
            self._validate_method_call(method_call)
            for definition, obj in self._spec['definitions'].items():
                locals()[definition] = obj
            result = eval(method_call)
            _debug(f'DEBUG: called method: {method_call}, result: {result}')
            return result or "Operation succeeded"
        except Exception as e:
            _debug(f'Exception running method: {method_call}: {e}')
            return f"The exception was caught calling {method_call}: {e}"

    def _validate_method_call(self, method):
        tree = ast.parse(method).body
        if len(tree) > 1:
            raise ProcessException("Incorrect function call")

    def _llm(self, messages):
        response = self.llm(messages=messages)
        _debug(f'assistant: {response}')
        return response


def _debug(message):
    if DEBUG:
        print(f'\033[2m{message}\033[0m')


def _parse_response(text):
    tags = _parse_tags(text)
    if not tags:
        return 'another', text
    return list(tags.items())[0]


def _parse_tags(text):
    # Pattern to match the structure: [TAG]content[/TAG]
    pattern = r"\<(\w+)\>(.*?)\<\/\1\>"
    # Find all matches and store them in a dictionary
    matches = re.findall(pattern, text, re.DOTALL)
    return {tag.lower(): content.strip() for tag, content in matches}


def _get_interface_md(specs):
    text = ""
    for signature, doc in specs['operations'].values():
        text += f"#### {signature}\n{doc}\n\n"
    if specs.get('models'):
        text += "### Models:\n"
        for model_name, schema in specs['models'].items():
            text += f"- {model_name}\n{schema}"

    return text


def _parse_interface(cls_or_module, prefix):
    operations = {}
    models = []
    models_dict = {}
    # model definitions are needed to set it in local() when generated
    # function call is executed.
    models_definitions = {}
    predicate = (
        inspect.isfunction if inspect.ismodule(cls_or_module)
        else inspect.ismethod
    )
    methods = inspect.getmembers(cls_or_module, predicate=predicate)
    do_methods = {
        name: func for name, func in methods
        if (name.startswith(prefix) if prefix else not name.startswith('_'))
    }

    for name, func in do_methods.items():
        doc = inspect.getdoc(func)
        signature = inspect.signature(func)
        operations[name] = (f"{name}{signature}", doc)

        for param_name, param_type in signature.parameters.items():
            if HAS_PYDANTIC:
                models = _parse_hints(param_type.annotation)

        models += _parse_hints(signature.return_annotation)

        for model in models:
            models_dict[model.__name__] = model.schema()
            models_definitions[model.__name__] = model

    return {'operations': operations, 'models': models_dict,
            'definitions': models_definitions}


def _parse_hints(annotation):
    models = []
    if hasattr(annotation, '__args__'):
        for arg in annotation.__args__:
            models += _parse_hints(arg)
    elif issubclass(annotation, BaseModel):
        models.append(annotation)
    return models


class OpenAIWrapper:
    def __init__(self, client, model):
        self.client = client
        self.model = model

    def __call__(self, messages):
        resp = self.client.chat.completions.create(
            messages=messages, model='gpt-4o'
        )
        return resp.choices[0].message.content


def _get_llm_wrapper(llm, model=None):
    if type(llm).__name__ == 'OpenAI':
        if model is None:
            raise Exception("Model not provided")
        return OpenAIWrapper(llm, model=model)
    else:
        return llm
