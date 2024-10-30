from ise.models.density_estimators.normalizing_flow import NormalizingFlow
from ise.models.ISEFlow import params

class ISEFlow_AIS_NF(NormalizingFlow):
    def __init__(self, ):
        self.input_size = params["AIS"]["input_size"] # 99
        self.output_size = params["AIS"]["output_size"] # 1
        self.num_flow_transforms = 5
        super().__init__(self, input_size=self.input_size, output_size=self.output_size, num_flow_transforms=self.num_flow_transforms)
        
class ISEFlow_GrIS_NF(NormalizingFlow):
    def __init__(self,):
        self.input_size = params["GrIS"]["input_size"] # 91
        self.output_size = params["GrIS"]["output_size"] # 1
        self.num_flow_transforms = 5
        super().__init__(self, input_size=self.input_size, output_size=self.output_size, num_flow_transforms=self.num_flow_transforms)