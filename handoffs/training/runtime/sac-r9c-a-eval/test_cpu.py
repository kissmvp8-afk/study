import unittest
from r4_protocol import command,analyze_rows
from evaluation_r9c_helpers import validate_seed
class Test(unittest.TestCase):
 def test_seeds(self):
  for x in (231,232,233,234,235,261,262,263,264,265):validate_seed('development',x,'standard')
  with self.assertRaises(AssertionError):validate_seed('development',271,'standard')
  with self.assertRaises(AssertionError):validate_seed('primary',271,'standard')
 def test_scoring(self):
  rows=[]
  for i in range(600):
   c=command('standard',i)
   rows.append(dict(step=i,t_start_s=i*.02,t_end_s=(i+1)*.02,effective_command=c,measured_body_vx_vy_wz=c,wheel_target_rad_s=[0.,0.],terminated=False,time_out=False,termination_terms={'base_contact':False,'time_out':False},external_torque_b_command=[[0.,0.,0.]],external_force_b_command=[[0.,0.,0.]],contact_named_max_N={'base':0.},root_pos_w=[0.,0.,.3]))
  self.assertTrue(analyze_rows(rows)['task_pass'])
  for row in rows[50:100]:row['measured_body_vx_vy_wz']=[.06,0.,0.]
  self.assertFalse(analyze_rows(rows)['task_pass'])
  short=rows[:29];short[-1]['terminated']=True;short[-1]['termination_terms']['base_contact']=True
  self.assertFalse(analyze_rows(short)['survived'])
if __name__=='__main__':unittest.main()
