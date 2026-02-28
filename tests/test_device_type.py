from unittest.mock import patch

import torch

from totalsegmentator.bin.TotalSegmentator import validate_device_type
from totalsegmentator.python_api import convert_device_to_string, select_device
import unittest
import argparse


class TestValidateDeviceType(unittest.TestCase):
    def test_valid_inputs(self):
        self.assertEqual(validate_device_type("gpu"), "gpu")
        self.assertEqual(validate_device_type("cpu"), "cpu")
        self.assertEqual(validate_device_type("mps"), "mps")
        self.assertEqual(validate_device_type("gpu:0"), "gpu:0")
        self.assertEqual(validate_device_type("gpu:1"), "gpu:1")

    def test_invalid_inputs(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            validate_device_type("invalid")
        with self.assertRaises(argparse.ArgumentTypeError):
            validate_device_type("gpu:invalid")
        with self.assertRaises(argparse.ArgumentTypeError):
            validate_device_type("gpu:-1")
        with self.assertRaises(argparse.ArgumentTypeError):
            validate_device_type("gpu:3.1415926")
        with self.assertRaises(argparse.ArgumentTypeError):
            validate_device_type("gpu:")

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=True)
    def test_select_device_prefers_mps_when_cuda_unavailable(self, _mps_available, _cuda_available):
        device = select_device("gpu")
        self.assertEqual(device, torch.device("mps"))

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=False)
    def test_select_device_falls_back_to_cpu_when_no_gpu_backend(self, _mps_available, _cuda_available):
        device = select_device("gpu")
        self.assertEqual(device, "cpu")

    @patch("torch.backends.mps.is_available", return_value=True)
    def test_select_device_accepts_explicit_mps(self, _mps_available):
        device = select_device("mps")
        self.assertEqual(device, torch.device("mps"))

    def test_convert_device_to_string_preserves_mps(self):
        self.assertEqual(convert_device_to_string(torch.device("mps")), "mps")
        self.assertEqual(convert_device_to_string(torch.device("cuda:0")), "gpu")


if __name__ == "__main__":
    unittest.main()
