"""TOOL: bake a compact, conservative Map11 navigation graph and flow table.

Graph paths are simulation routes, not Riot's undocumented pathfinder. Every
edge is checked against the pinned collision grid. 100-unit graph spacing is
an explicit approximation; narrow reachable passages may be rejected. There
is never an unchecked straight-line fallback. Run full bakes through Slurm.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from .navgrid import load_patch_map


def clear_segment(grid, start, end, radius, team=None):
    """Conservative sampled capsule, covering gaps with half-step inflation."""
    length = np.linalg.norm(np.asarray(end)-start)
    count = max(1, int(np.ceil(length / 10)))
    margin = length / count / 2
    return all(grid.is_walkable(float(p[0]), float(p[1]), radius=radius+margin, team=team)
               for p in np.linspace(start, end, count+1))


def build_routes(grid, out, *, spacing=100., radius=35.):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    xs = np.arange(grid.min_bounds[0]+spacing/2, grid.max_bounds[0], spacing)
    zs = np.arange(grid.min_bounds[2]+spacing/2, grid.max_bounds[2], spacing)
    # Both teams use a gate-closed graph. Runtime direct segments can still use
    # their team's gates; path planning across gates is an explicit limitation.
    cells = np.full((len(zs),len(xs)), -1, np.int32)
    points=[]
    for z,y in enumerate(zs):
        for x,v in enumerate(xs):
            if grid.is_walkable(v,y,radius=radius):
                cells[z,x]=len(points);points.append((v,y))
    points=np.asarray(points,np.float32)
    if len(points)>=32767: raise ValueError('graph too large for int16 routes')
    rows=[];cols=[];weights=[]
    for z,x in np.argwhere(cells>=0):
        i=cells[z,x]
        for dz,dx in ((0,1),(1,-1),(1,0),(1,1)):
            zz,xx=z+dz,x+dx
            if not (0<=zz<len(zs) and 0<=xx<len(xs)):continue
            j=cells[zz,xx]
            if j>=0 and clear_segment(grid,points[i],points[j],radius):
                dist=float(np.linalg.norm(points[i]-points[j]))
                rows.extend([i,j]);cols.extend([j,i]);weights.extend([dist,dist])
    graph=csr_matrix((weights,(rows,cols)),shape=(len(points),len(points)))
    paths=np.lib.format.open_memmap(out/'next.npy',mode='w+',dtype=np.int16,shape=(len(points),len(points)))
    for start in range(0,len(points),64):
        stop=min(start+64,len(points))
        _,pred=dijkstra(graph,directed=False,indices=np.arange(start,stop),return_predecessors=True)
        paths[start:stop]=np.where(pred<0,-1,pred).astype(np.int16)
    paths.flush()
    np.save(out/'points.npy',points);np.save(out/'cells.npy',cells)
    hashes={f:hashlib.sha256((out/f).read_bytes()).hexdigest() for f in ('points.npy','cells.npy','next.npy')}
    meta={'schema':'map11-flow-v1','patch':'26.19','spacing':spacing,'radius':radius,
          'min_x':grid.min_bounds[0],'min_z':grid.min_bounds[2],
          'grid_flags_sha256':hashlib.sha256(grid.flags.tobytes()).hexdigest(),
          'files':hashes,'nodes':len(points),'edges':len(rows),
          'limitations':['100-unit navigation graph','base gates closed in graph','static terrain only']}
    (out/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    return meta


def load_routes(path, grid):
    from ..map.pathing import FlowRoutes
    import jax.numpy as jnp
    path=Path(path);m=json.loads((path/'manifest.json').read_text())
    if m['schema']!='map11-flow-v1' or m['patch']!='26.19':raise ValueError('route profile mismatch')
    if m['grid_flags_sha256']!=hashlib.sha256(grid.flags.tobytes()).hexdigest():raise ValueError('route terrain mismatch')
    arrays={}
    for name in ('points','cells','next'):
        file=name+'.npy'
        if hashlib.sha256((path/file).read_bytes()).hexdigest()!=m['files'][file]:raise ValueError('route checksum mismatch')
        arrays[name]=jnp.asarray(np.load(path/file,allow_pickle=False))
    return FlowRoutes(arrays['points'],arrays['cells'],arrays['next'],m['spacing'],m['min_x'],m['min_z'],m['radius']),m


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--map',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    a=p.parse_args();g,_=load_patch_map(a.map)
    print(json.dumps(build_routes(g,a.out),indent=2))

if __name__=='__main__':main()
