from ise.models.density_estimators.normalizing_flow import NormalizingFlow

class ISEFlow_AIS_NF(NormalizingFlow):
    def __init__(self, ):
        self.input_size = 99
        self.output_size = 1
        self.num_flow_transforms = 5
        super().__init__(input_size=self.input_size, output_size=self.output_size, num_flow_transforms=self.num_flow_transforms)
        
class ISEFlow_GrIS_NF(NormalizingFlow):
    def __init__(self,):
        self.input_size = 90
        self.output_size = 1
        self.num_flow_transforms = 5
        super().__init__(input_size=self.input_size, output_size=self.output_size, num_flow_transforms=self.num_flow_transforms)