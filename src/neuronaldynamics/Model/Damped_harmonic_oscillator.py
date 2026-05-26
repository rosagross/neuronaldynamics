from Model.Models import General2DSystem
class Damped_harmonic_oscillator(General2DSystem):
    """
    Class that defines a damped harmonic oscillator as General2DSystem
    Set kwargs are model, model_name and parameters
    The latter two may be changed upon calling, the model is the intended version for the template
    """
    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        if 'model' not in kwargs:
            self.model = ['y', '-a*x - b*y']
        else:
            self.model = kwargs['model']

        if 'model_name' not in kwargs:
            self.model_name = 'damped harmonic oscillator'
        else:
            self.model_name = kwargs['model_name']

        if 'parameters' not in kwargs:
            self.parameters = {'a': 1.2, 'b': 1.0}
        else:
            self.parameters = kwargs['parameters']

