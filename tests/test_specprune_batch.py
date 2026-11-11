import unittest
from scripts.batch_specprune import aggregate

class SpecPruneBatchTests(unittest.TestCase):
 def row(self,t,i):
  return dict(task=t,trial=i,audit={'episodes':[dict(mode=m,success=True,task=t,trial=i) for m in ('baseline','accelerated')]})
 def test_counts_pairs_without_promoting_quality(self):
  r=aggregate([self.row(0,1),self.row(1,1)],2)
  self.assertEqual(r['successes'],{'baseline':2,'accelerated':2})
  self.assertFalse(r['deployment_accepted'])
 def test_missing_duplicate_and_wrong_task_are_rejected(self):
  for rows,n in (([self.row(0,1)],2),([self.row(0,1),self.row(0,1)],2)):
   with self.assertRaises(ValueError):aggregate(rows,n)
  r=self.row(0,1);r['audit']['episodes'][1]['task']=2
  with self.assertRaises(ValueError):aggregate([r],1)
 def test_query_latency_is_weighted_by_real_queries(self):
  rows=[self.row(0,5),self.row(1,5)]
  for i,row in enumerate(rows):
   for e in row['audit']['episodes']:
    e.update(policy_ms=100+100*i,policy_calls=8,environment_steps=7,
             model_calls=1+2*i,query_policy_ms=80+160*i)
  r=aggregate(rows,2)
  self.assertEqual(r['metrics']['accelerated']['mean_query_policy_ms'],80)
  self.assertEqual(r['metrics']['accelerated']['mean_episode_policy_ms'],150)
  self.assertEqual(r['metrics']['accelerated']['model_calls'],4)
