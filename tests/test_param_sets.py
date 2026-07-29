"""Tuned-parameter-set selection and the 15-parameter LhcBox."""

import unittest

import numpy as np

import cryoml.pdk_extract as px

PUBLISHED15 = {
    "vth0": 0.4119, "u0": 0.0740, "nfactor": 15.0, "vsat": 1.1e5,
    "delta": 0.02, "rdsw": 350.0, "eta0": 0.17,
    "pclm": 1.55436, "pdiblc1": 0.693725, "pdiblc2": 0.0608197,
    "ags": 0.4428, "ua": -9.55761e-11, "ub": 1.34494e-18,
    "voff": -0.2986804670, "prwg": 0.0,
}


class ActiveParams15(unittest.TestCase):
    """Run each test with ACTIVE_PARAMS swapped to the 15-parameter set."""

    def setUp(self):
        self._saved = px.ACTIVE_PARAMS
        px.ACTIVE_PARAMS = px.PARAMS15
        self.addCleanup(setattr, px, "ACTIVE_PARAMS", self._saved)


class ParamSetDefinitionTests(unittest.TestCase):
    def test_default_active_set_is_canonical_params7(self):
        self.assertEqual(px.ACTIVE_PARAM_SET, "params7")
        self.assertEqual(px.ACTIVE_PARAMS, px.PARAMS7)

    def test_params15_extends_params7_in_order(self):
        self.assertEqual(px.PARAMS15[:7], px.PARAMS7)
        self.assertEqual(len(px.PARAMS15), 15)
        self.assertEqual(len(set(px.PARAMS15)), 15)
        self.assertEqual(px.PARAM_SETS["params7"], px.PARAMS7)
        self.assertEqual(px.PARAM_SETS["params15"], px.PARAMS15)


class LhcBox15Tests(ActiveParams15):
    def test_box_is_ten_percent_and_sign_safe(self):
        box = px.LhcBox("nmos", 0, PUBLISHED15)
        for i, p in enumerate(px.PARAMS15):
            v = PUBLISHED15[p]
            if v == 0.0:
                continue
            self.assertAlmostEqual(box.lo[i], min(0.9 * v, 1.1 * v))
            self.assertAlmostEqual(box.hi[i], max(0.9 * v, 1.1 * v))
            self.assertLess(box.lo[i], box.hi[i])

    def test_z_roundtrip_15(self):
        box = px.LhcBox("nmos", 0, PUBLISHED15)
        z = np.linspace(-2.0, 2.0, 15)
        params = box.z_to_params(z)
        self.assertEqual(set(params), set(px.PARAMS15))
        z_back = box.params_to_z(params)
        nondeg = [i for i, p in enumerate(px.PARAMS15) if PUBLISHED15[p] != 0]
        np.testing.assert_allclose(z_back[nondeg], z[nondeg],
                                   rtol=1e-6, atol=1e-4)

    def test_zero_published_param_stays_pinned(self):
        box = px.LhcBox("nmos", 0, PUBLISHED15)
        i = px.PARAMS15.index("prwg")
        for z_val in (-8.0, 0.0, 8.0):
            z = np.zeros(15)
            z[i] = z_val
            self.assertLess(abs(box.z_to_params(z)["prwg"]), 1e-29)

    def test_z_published_recovers_published_values(self):
        box = px.LhcBox("nmos", 0, PUBLISHED15)
        params = box.z_to_params(box.z_published)
        for p in px.PARAMS15:
            if PUBLISHED15[p] == 0.0:
                continue
            self.assertAlmostEqual(params[p] / PUBLISHED15[p], 1.0, places=4)

    def test_wide_box_rejected_outside_params7(self):
        with self.assertRaises(ValueError):
            px.make_box("nmos", 0.15, 1.6, 0, mode="wide")


if __name__ == "__main__":
    unittest.main()
