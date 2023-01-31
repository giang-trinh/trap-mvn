import mpbn
import sys


model = sys.argv[1]
computation = sys.argv[2]
max_output = int(sys.argv[3])
mbn = mpbn.load(model)
if computation == "fix":
    tspaces = mbn.fixedpoints(limit=max_output)
else:
    tspaces = mbn.attractors(limit=max_output)
tspaces = list(tspaces)
for s in tspaces:
    print(s)