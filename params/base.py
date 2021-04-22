from dataclasses import dataclass


class ValidationError(Exception):
    def __init__(self, *args, **kwargs):
        if not args and not kwargs:
            super().__init__("Parameter validation failed")
        else:
            super().__init__(*args, **kwargs)


@dataclass
class Parameter:
    id: str


@dataclass
class NumberParameter(Parameter):
    value: float

    @classmethod
    def validate(kls, value: float):
        try:
            float(value)
        except ValueError:
            raise ValidationError()


@dataclass
class BoolParameter(Parameter):
    value: bool


@dataclass
class StringParameter(Parameter):
    value: str
