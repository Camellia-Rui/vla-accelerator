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
