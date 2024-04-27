class ErrorResponse():

    def __init__(
        self,
        message: str,
    ) -> None:
        self.message = message

    def to_json(self):
        return self.__dict__


class GeneralResponse():

    def __init__(
        self,
        message: str,
    ) -> None:
        self.message = message

    def to_json(self):
        return self.__dict__


class AnswerResponse():

    def __init__(
        self,
        answer: str,
    ) -> None:
        self.answer = answer

    def to_json(self):
        return self.__dict__
