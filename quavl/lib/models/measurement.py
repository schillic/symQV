from typing import List, Union

from quavl.lib.expressions.qbit import QbitVal


class Measurement:
    def __init__(self, arguments: Union[QbitVal, List[QbitVal]]):
        self.arguments = arguments

    def __repr__(self):
        if isinstance(self.arguments, List):
            return f'measure{[q.get_identifier() for q in self.arguments]}'

        return f'measure {self.arguments.get_identifier()}'
