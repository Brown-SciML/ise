from ise.models.density_estimators.normalizing_flow import NormalizingFlow

class ISEFlow_AIS_NF(NormalizingFlow):
    """
    ISEFlow_AIS_NF is a specialized normalizing flow model for the Antarctic Ice Sheet (AIS).

    This class extends the `NormalizingFlow` class, configuring it with AIS-specific input sizes and transformations.
    """

    def __init__(self, ):
        """
        Initializes the ISEFlow_AIS_NF model.

        This model is pre-configured with:
        - `input_size` of 99, representing the number of input features.
        - `output_size` of 1, representing the target variable.
        - `num_flow_transforms` of 5, specifying the number of flow transformations.

        Calls the `NormalizingFlow` constructor with these preset parameters.
        """

        self.input_size = 99
        self.output_size = 1
        self.num_flow_transforms = 5
        super().__init__(input_size=self.input_size, output_size=self.output_size, num_flow_transforms=self.num_flow_transforms)
        
class ISEFlow_GrIS_NF(NormalizingFlow):
    """
    ISEFlow_GrIS_NF is a specialized normalizing flow model for the Greenland Ice Sheet (GrIS).

    This class extends the `NormalizingFlow` class, configuring it with GrIS-specific input sizes and transformations.
    """

    def __init__(self,):
        """
        Initializes the ISEFlow_GrIS_NF model.

        This model is pre-configured with:
        - `input_size` of 90, representing the number of input features.
        - `output_size` of 1, representing the target variable.
        - `num_flow_transforms` of 5, specifying the number of flow transformations.

        Calls the `NormalizingFlow` constructor with these preset parameters.
        """

        self.input_size = 90
        self.output_size = 1
        self.num_flow_transforms = 5
        super().__init__(input_size=self.input_size, output_size=self.output_size, num_flow_transforms=self.num_flow_transforms)