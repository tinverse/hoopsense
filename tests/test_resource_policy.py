import os
import unittest
from unittest import mock

from pipelines.resource_policy import apply_cpu_thread_policy, default_cpu_thread_count


class ResourcePolicyTest(unittest.TestCase):
    def test_default_cpu_thread_count_is_conservative(self):
        self.assertEqual(default_cpu_thread_count(1), 1)
        self.assertEqual(default_cpu_thread_count(4), 4)
        self.assertEqual(default_cpu_thread_count(32), 4)

    def test_apply_cpu_thread_policy_preserves_explicit_env(self):
        with mock.patch.dict(os.environ, {"OMP_NUM_THREADS": "2"}, clear=True):
            limits = apply_cpu_thread_policy(4)
        self.assertEqual(limits["OMP_NUM_THREADS"], "2")
        self.assertEqual(limits["OPENBLAS_NUM_THREADS"], "4")
        self.assertEqual(limits["MKL_NUM_THREADS"], "4")
        self.assertEqual(limits["NUMEXPR_NUM_THREADS"], "4")


if __name__ == "__main__":
    unittest.main()
