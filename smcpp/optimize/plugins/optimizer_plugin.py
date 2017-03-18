from smcpp.observe import Observer, targets

class OptimizerPlugin(Observer):
    'Abstract class used only to track subclasses'
    DISABLED = False
