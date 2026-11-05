import unittest
from types import SimpleNamespace
from methods.specprune.lifecycle import ModelSnapshot

class SpecPruneLifecycleTests(unittest.TestCase):
 def test_restores_modules_and_mutable_state_without_alias_mutation(self):
  attn=object(); state={'old':3}
  dec=SimpleNamespace(layers=[SimpleNamespace(self_attn=attn)],forward=object(),dynamic_importance_confidence=state)
  model=SimpleNamespace(language_model=SimpleNamespace(model=dec),predict_action=object())
  before=vars(dec).copy(); predict=model.predict_action
  snap=ModelSnapshot(model); snap.isolate()
  dec.dynamic_importance_confidence['new']=4
  dec.layers[0].self_attn=object(); model.predict_action=object(); model._orig_predict_action=predict
  snap.restore(); snap.restore()
  self.assertIs(dec.layers[0].self_attn,attn)
  self.assertIs(dec.dynamic_importance_confidence,state)
  self.assertEqual(state,{'old':3})
  self.assertEqual(set(vars(dec)),set(before))
  self.assertIs(model.predict_action,predict)
  self.assertFalse(hasattr(model,'_orig_predict_action'))

 def test_class_methods_are_not_left_as_instance_overrides(self):
  class Model:
   def predict_action(self): return 1
  model=Model(); model.language_model=SimpleNamespace(model=SimpleNamespace(layers=[],forward=object()))
  snap=ModelSnapshot(model)
  model.predict_action=lambda:2
  snap.restore()
  self.assertNotIn('predict_action',vars(model))
  self.assertEqual(model.predict_action(),1)

class SpecPruneRoutingTests(unittest.TestCase):
 def test_evaluate_route_and_deployment_rejection(self):
  from core.run import main
  from unittest.mock import patch
  with patch('methods.specprune.evaluation.main',return_value=0) as run:
   self.assertEqual(main(['--method','specprune','--help'],evaluate=True),0)
   run.assert_called_once_with(['--help'])
  with self.assertRaises(SystemExit): main(['--method','specprune'])

 def test_invalid_protocol_stops_before_model_loading(self):
  from methods.specprune.evaluation import main
  with self.assertRaises(SystemExit):
   main(['--workspace','.','--checkpoint','.','--observation','.','--output','unused','--task','10'])
