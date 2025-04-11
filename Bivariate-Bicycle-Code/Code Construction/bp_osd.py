from ldpc.bposd_decoder import BpOsdDecoder
import numpy as np
from dem_to_matrices import detector_error_model_to_check_matrices

import stim

DEFAULT_MAX_BP_ITERS = 30
DEFAULT_BP_METHOD = "product_sum"
DEFAULT_OSD_ORDER = 60
DEFAULT_OSD_METHOD = "osd_cs"

class BPOSD:
    def __init__(
        self,
        model: stim.DetectorErrorModel,
        max_bp_iters: int = DEFAULT_MAX_BP_ITERS,
        bp_method: str = DEFAULT_BP_METHOD,
        osd_order: int = DEFAULT_OSD_ORDER,
        osd_method: str = DEFAULT_OSD_METHOD,
        **bposd_kwargs,
    ):
        f"""Class for decoding stim circuits using belief propagation and ordered statistics decoding (BP+OSD).
        This class uses Joschka Roffe's BP+OSD decoder as a subroutine. For more information on the options and 
        implementation of the BP+OSD subroutine, see the documentation of the LDPC library: https://roffe.eu/software/ldpc/index.html.
        Additional keyword arguments are passed to the ``bposd_decoder`` class of the ldpc Python package.

        Parameters
        ----------
        model : stim.DetectorErrorModel
            The detector error model of the stim circuit to be decoded
        max_bp_iters : int, optional
            The maximum number of iterations of belief propagation to be used, by default {DEFAULT_MAX_BP_ITERS}
        bp_method : str, optional
            The BP method. Currently three methods are implemented: 1) "product_sum": product sum updates;
            2) "min_sum": min-sum updates; 3) "min_sum_log": min-sum log updates, by default {DEFAULT_BP_METHOD}
        osd_order : int, optional
            The OSD order, by default {DEFAULT_OSD_ORDER}
        osd_method : str, optional
            The OSD method. Currently three methods are available: 1) "osd_0": Zero-oder OSD; 2) "osd_e": 
            exhaustive OSD; 3) "osd_cs": combination-sweep OSD., by default {DEFAULT_OSD_METHOD}
        """
        self._matrices = detector_error_model_to_check_matrices(
            model, allow_undecomposed_hyperedges=True
        )
        self.num_detectors = model.num_detectors
        self.num_errors = model.num_errors
        h_shape = self._matrices.check_matrix.shape
        # print("Check matrix:", self._matrices.check_matrix)
        # print("Priors:", self._matrices.priors)
        print("Check matrix shape:", self._matrices.check_matrix.shape)
        # print("Priors shape:", self._matrices.priors.shape)
        max_osd_order = h_shape[1] - h_shape[0]
        if max_osd_order <= 0:
            max_osd_order = 1
        self._bposd = BpOsdDecoder(
            pcm=self._matrices.check_matrix,
            max_iter=max_bp_iters,
            bp_method=bp_method,
            error_channel=list(self._matrices.priors),
            osd_order=min(osd_order, max_osd_order),
            osd_method=osd_method,
            input_vector_type="syndrome",
            **bposd_kwargs,
        )

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """
        Decode the syndrome and return a prediction of which observables were flipped

        Parameters
        ----------
        syndrome : np.ndarray
            A single shot of syndrome data. This should be a binary array with a length equal to the
            number of detectors in the `stim.Circuit` or `stim.DetectorErrorModel`. E.g. the syndrome might be
            one row of shot data sampled from a `stim.CompiledDetectorSampler`.

        Returns
        -------
        np.ndarray
            A binary numpy array `predictions` which predicts which observables were flipped.
            Its length is equal to the number of observables in the `stim.Circuit` or `stim.DetectorErrorModel`.
            `predictions[i]` is 1 if the decoder predicts observable `i` was flipped and 0 otherwise.
        """
        corr = self._bposd.decode(syndrome)
        return (self._matrices.observables_matrix @ corr) % 2

    def decode_batch(
        self,
        shots: np.ndarray,
        *,
        bit_packed_shots: bool = False,
        bit_packed_predictions: bool = False,
    ) -> np.ndarray:
        """
        Decode a batch of shots of syndrome data. This is just a helper method, equivalent to iterating over each
        shot and calling `BPOSD.decode` on it.

        Parameters
        ----------
        shots : np.ndarray
            A binary numpy array of dtype `np.uint8` or `bool` with shape `(num_shots, num_detectors)`, where
            here `num_shots` is the number of shots and `num_detectors` is the number of detectors in the `stim.Circuit` or `stim.DetectorErrorModel`.

        Returns
        -------
        np.ndarray
            A 2D numpy array `predictions` of dtype bool, where `predictions[i, :]` is the output of
            `self.decode(shots[i, :])`.
        """
        if bit_packed_shots:
            shots = np.unpackbits(shots, axis=1, bitorder="little")[
                :, : self.num_detectors
            ]
        predictions = np.zeros(
            (shots.shape[0], self._matrices.observables_matrix.shape[0]), dtype=bool
        )
        for i in range(shots.shape[0]):
            predictions[i, :] = self.decode(shots[i, :])
        if bit_packed_predictions:
            predictions = np.packbits(predictions, axis=1, bitorder="little")
        return predictions
