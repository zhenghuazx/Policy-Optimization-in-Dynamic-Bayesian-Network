from abc import ABC,abstractmethod


class base_gradient(ABC):
    @abstractmethod
    def func(self, theta):
        pass

    @abstractmethod
    def grad_f(self, theta):
        pass