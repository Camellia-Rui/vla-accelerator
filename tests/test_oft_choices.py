from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
from methods.oft_choices import OFTChoice


class OFTChoiceTests(unittest.TestCase):
    def setUp(self):
        self.bundle=dict(model=object(),native_reset=Mock(),native_query=Mock(return_value=[[1.0]*7]),
                         processor=object(),action_head=object(),proprio_projector=object())
        self.choice=OFTChoice(self.bundle,object())

    def test_switch_detaches_old_adapter_before_installing_new(self):
        old,new=Mock(),Mock()
        with patch("methods.lightvla.LightVLAAccelerator.install",return_value=old), \
             patch("methods.vla_pruner.accelerator.VLAPrunerAccelerator.install",return_value=new) as install:
            self.choice.select("lightvla")
            def check(*args,**kwargs):
                old.detach.assert_called_once()
                return new
            install.side_effect=check
            self.choice.select("vla_pruner")
            self.choice.close()
        new.detach.assert_called_once()
        self.assertEqual(self.choice.mode,"baseline")

    def test_failed_install_leaves_native_mode_and_unknown_does_not_mutate(self):
        with patch("methods.lightvla.LightVLAAccelerator.install",side_effect=ValueError("incompatible")):
            with self.assertRaises(ValueError):self.choice.select("lightvla")
        self.assertIsNone(self.choice.adapter)
        self.assertEqual(self.choice.mode,"baseline")
        self.bundle["native_reset"].reset_mock()
        with self.assertRaises(ValueError):self.choice.select("unknown")
        self.bundle["native_reset"].assert_not_called()

    def test_baseline_preserves_complete_action_chunk(self):
        self.bundle["native_query"].return_value=[[float(i)]*7 for i in range(8)]
        self.assertEqual(self.choice.query({}).shape,(8,7))
