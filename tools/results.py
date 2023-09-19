import mmengine


pth = 'work_dirs/rn50_1/out.pkl'


preds = mmengine.load(pth)

print('dbg')