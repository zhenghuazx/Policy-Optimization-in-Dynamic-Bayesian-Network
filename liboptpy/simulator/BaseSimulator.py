from abc import ABC,abstractmethod

class BaseSimulator(ABC):
    @abstractmethod
    def f(self, xs, t, ps):
        pass

    @abstractmethod
    def g(self, xs, t, ps):
        pass