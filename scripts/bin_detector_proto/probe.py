import sys, math, json
import data, detectors as D
cams = data.cameras(); P = {p['label']: p for p in data.png_items()}; V = {v['label']: v for v in data.vid_items()}
method = sys.argv[1]; labs = sys.argv[2].split(',')
params = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
model = None
if method == 'learn':
    import pickle; model = pickle.load(open(params.pop('model'), 'rb'))
det = D.Detector(method, cams, model, params)
for lab in labs:
    it = P.get(lab) or V.get(lab); diag = {}
    r = det.localize(data.load(it), it['exclude'], diag=diag)
    sx, sy = data.SPOTS.get(it['spot'], (9, 9))
    print(lab, it['spot'], {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, 'd=%.3f' % math.hypot(r.get('x', 99) - sx, r.get('y', 99) - sy))
    print('   pool', [(round(a, 3), b, round(c, 2), round(d, 2), round(math.degrees(e))) for a, b, c, d, e in diag['pool'][:5]])
