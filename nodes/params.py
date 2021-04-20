from dataclasses import dataclass


@dataclass
class Parameter:
    id: str


@dataclass
class IntParameter(Parameter):
    value: int

    def to_python(self):
        return self.value


@dataclass
class BoolParameter(Parameter):
    value: bool
