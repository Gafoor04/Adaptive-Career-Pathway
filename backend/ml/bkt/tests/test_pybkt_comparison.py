"""Comparative validation tests against the pyBKT library.

Validates that BKTEngine matches pyBKT's forward state transitions and prediction
equations when evaluated using identical parameter values and interaction sequences.
"""

import math
import numpy as np
import pandas as pd
import pytest

pyBKT = pytest.importorskip("pyBKT")
from pyBKT.models import Model

from ml.bkt.parameters import BKTParameters
from ml.bkt.model import BKTEngine


class TestPyBKTComparison:
    """Cross-engine numerical comparison between custom BKTEngine and pyBKT."""

    def test_forward_predictions_match_pybkt_equations(self):
        """Verify BKTEngine produces mathematically identical predictions to pyBKT.

        We fit a small reference sequence in pyBKT, extract pyBKT's learned parameters,
        and then run BKTEngine over the identical sequence to assert equivalence of:
            1. Step-level prior mastery states (pyBKT 'state_predictions')
            2. Step-level response predictions (pyBKT 'correct_predictions')
        """
        # Create reference sequence DataFrame
        responses = [1, 0, 1, 1, 0]
        n_steps = len(responses)
        df = pd.DataFrame({
            "user_id": ["learner_01"] * n_steps,
            "skill_name": ["concept_alpha"] * n_steps,
            "correct": responses,
        })

        # Fit pyBKT Model to obtain a valid parameter set
        pybkt_model = Model(seed=42)
        pybkt_model.fit(data=df)

        # Extract pyBKT parameters
        raw_params = pybkt_model.coef_["concept_alpha"]
        prior = float(raw_params["prior"])
        learns = float(raw_params["learns"][0])
        guesses = float(raw_params["guesses"][0])
        slips = float(raw_params["slips"][0])
        forgets = float(raw_params["forgets"][0])

        # Get pyBKT predictions
        pybkt_preds = pybkt_model.predict(data=df)
        pybkt_p_correct = pybkt_preds["correct_predictions"].to_numpy()
        pybkt_states = pybkt_preds["state_predictions"].to_numpy()

        # Build custom BKTParameters from pyBKT values
        # (Allow p_guess/p_slip without monotonicity check here if pyBKT unconstrained fit inverted them)
        custom_params = BKTParameters.model_construct(
            p_l0=prior,
            p_transition=learns,
            p_guess=guesses,
            p_slip=slips,
            p_forget=forgets,
            version="1.0.0",
            source="pybkt_extracted",
        )

        # Step through BKTEngine with identical sequence
        current_mastery = prior
        for step_idx in range(n_steps):
            obs = bool(responses[step_idx])

            # 1. Compare prior state against pyBKT 'state_predictions'
            expected_state = float(pybkt_states[step_idx])
            assert math.isclose(current_mastery, expected_state, rel_tol=1e-4, abs_tol=1e-5), (
                f"Step {step_idx}: Mastery state mismatch. BKTEngine={current_mastery:.5f}, pyBKT={expected_state:.5f}"
            )

            # 2. Compare prediction against pyBKT 'correct_predictions'
            custom_p_pred = BKTEngine.predict_correct_probability(current_mastery, custom_params)
            expected_p_pred = float(pybkt_p_correct[step_idx])
            assert math.isclose(custom_p_pred, expected_p_pred, rel_tol=1e-4, abs_tol=1e-5), (
                f"Step {step_idx}: Prediction mismatch. BKTEngine={custom_p_pred:.5f}, pyBKT={expected_p_pred:.5f}"
            )

            # 3. Advance state via BKTEngine step
            next_mastery, _ = BKTEngine.step(
                current_mastery=current_mastery,
                is_correct=obs,
                params=custom_params,
                step_index=step_idx + 1,
            )
            current_mastery = next_mastery

    def test_parameter_roundtrip_dict_conversion(self):
        """Verify serialization fidelity between BKTParameters and pyBKT parameter dict."""
        params = BKTParameters(
            p_l0=0.18,
            p_transition=0.12,
            p_guess=0.22,
            p_slip=0.08,
            p_forget=0.02,
        )
        pybkt_dict = params.to_pybkt_dict()

        assert pybkt_dict == {
            "prior": 0.18,
            "learns": 0.12,
            "guesses": 0.22,
            "slips": 0.08,
            "forgets": 0.02,
        }

        reconstructed = BKTParameters.from_pybkt_dict(pybkt_dict)
        assert reconstructed.p_l0 == params.p_l0
        assert reconstructed.p_transition == params.p_transition
        assert reconstructed.p_guess == params.p_guess
        assert reconstructed.p_slip == params.p_slip
        assert reconstructed.p_forget == params.p_forget
