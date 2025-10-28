
from flask import Flask, render_template, request, redirect, url_for
import json
import numpy as np

app = Flask(__name__)

EPS0 = 8.8541878128e-12
MU0  = 4e-7 * np.pi

def parse_float(s, default=0.0):
    try:
        return float(str(s).replace(",", "."))
    except Exception:
        return default

def E_field_at_point(x, y, charges):
    ex, ey = 0.0, 0.0
    for c in charges:
        q = c["q"]; (x0, y0) = c["pos"]
        dx = x - x0; dy = y - y0
        r2 = dx*dx + dy*dy
        if r2 < 1e-9:  # avoid singularity
            continue
        r = r2 ** 0.5
        k = q / (4.0 * np.pi * EPS0 * r2 * r)
        ex += k * dx; ey += k * dy
    return ex, ey

def e_streamlines(charges, domain=2.0, n_seeds=56, step=0.015, max_steps=3000):
    seeds = []
    for c in charges:
        (x0, y0) = c["pos"]
        r = 0.12
        angles = np.linspace(0, 2*np.pi, max(8, n_seeds//max(1,len(charges))), endpoint=False)
        for a in angles: seeds.append([x0 + r*np.cos(a), y0 + r*np.sin(a)])
    lines = []
    x_min = y_min = -domain; x_max = y_max = +domain
    for sx, sy in seeds:
        for direction in (+1, -1):
            x, y = sx, sy; line = [[x, y]]
            for _ in range(max_steps):
                ex, ey = E_field_at_point(x, y, charges)
                e = (ex*ex + ey*ey) ** 0.5
                if e < 1e-16: break
                ex /= e; ey /= e
                mx = x + direction*step*ex*0.5; my = y + direction*step*ey*0.5
                mex, mey = E_field_at_point(mx, my, charges)
                me = (mex*mex + mey*mey) ** 0.5
                if me < 1e-16: break
                mex /= me; mey /= me
                x += direction*step*mex; y += direction*step*mey
                if x < x_min-0.05 or x > x_max+0.05 or y < y_min-0.05 or y > y_max+0.05: break
                close = any((x-c["pos"][0])**2 + (y-c["pos"][1])**2 < 0.02**2 for c in charges)
                if close: break
                line.append([x, y])
            if len(line) > 4: lines.append(line)
    return lines

def B_field_at_point(x, y, wires):
    bx, by = 0.0, 0.0
    for w in wires:
        I = w["I"]; (x0, y0) = w["pos"]
        dx = x - x0; dy = y - y0
        r2 = dx*dx + dy*dy
        if r2 < 1e-8: continue
        k = MU0 * I / (2*np.pi*r2)
        bx += -k * dy; by +=  k * dx
    return bx, by

def b_streamlines(wires, domain=2.0, n_seeds=32, step=0.015, max_steps=3000):
    seeds = []
    for w in wires:
        (x0, y0) = w["pos"]
        r = 0.12
        angles = np.linspace(0, 2*np.pi, max(6, n_seeds//max(1,len(wires))), endpoint=False)
        for a in angles: seeds.append([x0 + r*np.cos(a), y0 + r*np.sin(a)])
    lines = []
    x_min = y_min = -domain; x_max = y_max = +domain
    for sx, sy in seeds:
        for direction in (+1, -1):
            x, y = sx, sy; line = [[x, y]]
            for _ in range(max_steps):
                bx, by = B_field_at_point(x, y, wires)
                b = (bx*bx + by*by) ** 0.5
                if b < 1e-12: break
                bx /= b; by /= b
                mx = x + direction*step*bx*0.5; my = y + direction*step*by*0.5
                mbx, mby = B_field_at_point(mx, my, wires)
                mb = (mbx*mbx + mby*mby) ** 0.5
                if mb < 1e-12: break
                mbx /= mb; mby /= mb
                x += direction*step*mbx; y += direction*step*mby
                if x < x_min-0.05 or x > x_max+0.05 or y < y_min-0.05 or y > y_max+0.05: break
                close = any((x-w["pos"][0])**2 + (y-w["pos"][1])**2 < 0.02**2 for w in wires)
                if close: break
                line.append([x, y])
            if len(line) > 4: lines.append(line)
    return lines

def potential_at_point(x, y, charges, soft=1e-4):
    V = 0.0
    for c in charges:
        q = c["q"]; (x0, y0) = c["pos"]
        dx = x - x0; dy = y - y0
        r = (dx*dx + dy*dy + soft*soft) ** 0.5
        V += q / (4.0 * np.pi * EPS0 * r)
    return V

def grid_potential(xmin, xmax, ymin, ymax, nx, ny, charges):
    xs = np.linspace(xmin, xmax, nx); ys = np.linspace(ymin, ymax, ny)
    Z = np.zeros((ny, nx), dtype=float)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            Z[i, j] = potential_at_point(x, y, charges)
    return xs, ys, Z

def interp(p1, p2, v1, v2, level):
    if (v2 - v1) == 0: t = 0.5
    else: t = (level - v1) / (v2 - v1)
    return (p1[0] + t*(p2[0]-p1[0]), p1[1] + t*(p2[1]-p1[1]))

CASE_SEGMENTS = {0:[],15:[],1:[(3,0)],14:[(0,3)],2:[(0,1)],13:[(1,0)],
                 3:[(3,1)],12:[(1,3)],4:[(1,2)],11:[(2,1)],
                 5:[(3,2),(0,1)],10:[(0,3),(1,2)],6:[(0,2)],9:[(2,0)],7:[(3,2)],8:[(2,3)]}

def marching_squares(xs, ys, Z, levels):
    nx = len(xs); ny = len(ys)
    lines_by_level = {float(l): [] for l in levels}
    for level in levels:
        segs = []
        for i in range(ny-1):
            for j in range(nx-1):
                bl = Z[i, j]; br = Z[i, j+1]; tr = Z[i+1, j+1]; tl = Z[i+1, j]
                idx = 0
                if bl >= level: idx |= 1
                if br >= level: idx |= 2
                if tr >= level: idx |= 4
                if tl >= level: idx |= 8
                if idx == 0 or idx == 15: continue
                x0, x1 = xs[j], xs[j+1]; y0, y1 = ys[i], ys[i+1]
                epts = {0:interp((x0,y0),(x1,y0), bl, br, level),
                        1:interp((x1,y0),(x1,y1), br, tr, level),
                        2:interp((x1,y1),(x0,y1), tr, tl, level),
                        3:interp((x0,y1),(x0,y0), tl, bl, level)}
                for a,b in CASE_SEGMENTS[idx]: segs.append((epts[a], epts[b]))
        from collections import defaultdict
        def key(p): return (round(p[0], 4), round(p[1], 4))
        unused = set(range(len(segs)))
        adjacency = defaultdict(list)
        for idx, (a,b) in enumerate(segs):
            adjacency[key(a)].append(("out", idx)); adjacency[key(b)].append(("in", idx))
        polylines = []
        while unused:
            idx0 = unused.pop(); a,b = segs[idx0]; line = [a, b]
            cur = b
            while True:
                kcur = key(cur)
                nexts = [i for (d,i) in adjacency[kcur] if d=="out" and i in unused]
                if not nexts: break
                nxt = nexts[0]; unused.remove(nxt); _, nb = segs[nxt]; line.append(nb); cur = nb
            cur = a
            while True:
                kcur = key(cur)
                prevs = [i for (d,i) in adjacency[kcur] if d=="in" and i in unused]
                if not prevs: break
                prv = prevs[0]; unused.remove(prv); pa,_ = segs[prv]; line.insert(0, pa); cur = pa
            if len(line) >= 4: polylines.append(line)
        lines_by_level[float(level)] = polylines
    return lines_by_level

@app.route("/")
def index():
    return redirect(url_for("eletrico"))

@app.route("/eletrico", methods=["GET","POST"])
def eletrico():
    form = dict(q1="1e-9", x1="-0.6", y1="0.0", q2="-1e-9", x2="0.6", y2="0.0")
    if request.method == "POST":
        for k in list(form.keys()): form[k] = request.form.get(k, form[k])
    charges = [
        {"q": parse_float(form["q1"], 1e-9), "pos": (parse_float(form["x1"], -0.6), parse_float(form["y1"], 0.0))},
        {"q": parse_float(form["q2"],-1e-9), "pos": (parse_float(form["x2"],  0.6), parse_float(form["y2"], 0.0))},
    ]
    stream = e_streamlines(charges, domain=2.0, n_seeds=56)
    payload = dict(streamlines=stream, charges=[{"q": c["q"], "pos": c["pos"]} for c in charges])
    return render_template("sim_eletrico.html", form_data=form, simulation_data=json.dumps(payload), active_tab="eletrico", title="Simulador • Campo Elétrico — VS Code Dark")

@app.route("/magnetico", methods=["GET","POST"])
def magnetico():
    form = dict(I1="2.0", x1="-0.6", y1="0.0", I2="-2.0", x2="0.6", y2="0.0")
    if request.method == "POST":
        for k in list(form.keys()): form[k] = request.form.get(k, form[k])
    wires = [
        {"I": parse_float(form["I1"], 2.0), "pos": (parse_float(form["x1"], -0.6), parse_float(form["y1"], 0.0))},
        {"I": parse_float(form["I2"],-2.0), "pos": (parse_float(form["x2"],  0.6), parse_float(form["y2"], 0.0))},
    ]
    stream = b_streamlines(wires, domain=2.0, n_seeds=32)
    payload = dict(streamlines=stream, wires=[{"I": w["I"], "pos": w["pos"]} for w in wires])
    return render_template("sim_magnetico.html", form_data=form, simulation_data=json.dumps(payload), active_tab="magnetico", title="Simulador • Campo Magnético — VS Code Dark")

@app.route("/equipotencial", methods=["GET","POST"])
def equipotencial():
    form = dict(q1="1e-9", x1="-0.6", y1="0.0", q2="-1e-9", x2="0.6", y2="0.0", nlevels="14")
    if request.method == "POST":
        for k in list(form.keys()): form[k] = request.form.get(k, form[k])
    charges = [
        {"q": parse_float(form["q1"], 1e-9), "pos": (parse_float(form["x1"], -0.6), parse_float(form["y1"], 0.0))},
        {"q": parse_float(form["q2"],-1e-9), "pos": (parse_float(form["x2"],  0.6), parse_float(form["y2"], 0.0))},
    ]
    xs, ys, Z = grid_potential(-2, 2, -2, 2, 81, 81, charges)
    vmax = np.quantile(np.abs(Z), 0.90)
    levels_cnt = max(6, min(30, int(parse_float(form["nlevels"], 14))))
    levels = np.linspace(-0.9*vmax, 0.9*vmax, levels_cnt)
    contours = marching_squares(xs, ys, Z, levels)
    payload = dict(
        equipotentials=[{"level": float(L), "line": poly} for L, lines in contours.items() for poly in lines],
        charges=[{"q": c["q"], "pos": c["pos"]} for c in charges]
    )
    return render_template("sim_equipotencial.html", form_data=form, simulation_data=json.dumps(payload), active_tab="equipotencial", title="Simulador • Equipotenciais — VS Code Dark")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
