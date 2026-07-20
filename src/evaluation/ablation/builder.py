from models.hybrid_model import HybridProteinClassifier


class HybridModelBuilder:
    """
    Builds HQNN architectures for ablation studies.
    """

    def __init__(self, configuration):

        self.configuration = configuration

    def build(self, variant):

        model = HybridProteinClassifier(

            cnn_enabled=variant.cnn,

            quantum_enabled=variant.quantum,

            multiscale_enabled=variant.multiscale,

            residual_enabled=variant.residual,

            data_reuploading_enabled=variant.data_reuploading,

            entanglement_enabled=variant.entanglement,

            quantum_layers=variant.vqc_layers,

        )

        return model