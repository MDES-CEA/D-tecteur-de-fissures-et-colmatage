"""Deterministic centerline union with an explicit spatial tolerance.

Only parallel (within five degrees) longitudinally overlapping intervals merge.
Crossings and endpoint contacts do not merge. First trace ID supplies geometry;
disagreeing labels on shared intervals become conflict. No annotations mutate.
"""
from collections import defaultdict
from math import ceil, floor, hypot, atan2, degrees


def distribute_lengths(segments, zones):
    """Partition the already deduplicated geometry into disjoint tile bins."""
    from bisect import bisect_right
    totals={z['id']:0. for z in zones}
    xs=sorted({v for z in zones for v in (z['x'],z['x']+z['width'])})
    ys=sorted({v for z in zones for v in (z['y'],z['y']+z['height'])})
    lookup={(z['x'],z['y']):z['id'] for z in zones}
    for segment in segments:
        a,b=segment['geometry'];dx,dy=b[0]-a[0],b[1]-a[1]
        cuts={0.,1.}
        for origin,delta,edges in ((a[0],dx,xs),(a[1],dy,ys)):
            if abs(delta)<1e-10: continue
            low,high=sorted((origin,origin+delta))
            for edge in edges[bisect_right(edges,low):bisect_right(edges,high)]:
                t=(edge-origin)/delta
                if 0<t<1:cuts.add(t)
        cuts=sorted(cuts)
        for lo,hi in zip(cuts,cuts[1:]):
            t=(lo+hi)/2;x=a[0]+t*dx;y=a[1]+t*dy
            ix=bisect_right(xs,x)-1;iy=bisect_right(ys,y)-1
            if ix<0 or iy<0:continue
            zone_id=lookup.get((xs[ix],ys[iy]))
            if zone_id is not None:totals[zone_id]+=(hi-lo)*segment['length_px']
    return totals


def measure_union(cracks, tolerance_px=0):
    from .length_analysis import CATEGORIES
    if not 0 <= tolerance_px <= 5:
        raise ValueError('Tolerance must be between 0 and 5 pixels')
    active = [c for c in cracks if c.get('state') != 'rejected' and len(c['geometry']) >= 2]
    active.sort(key=lambda c: (c.get('id', 0), str(c['geometry'])))
    parents = list(range(len(active)))
    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]; i = parents[i]
        return i
    def join(i,j):
        parents[root(j)] = root(i)
    grid = defaultdict(list)
    accepted = []
    raw = 0.0
    valid_owners = set()
    def cells(a,b,padding=0):
        for x in range(floor((min(a[0],b[0])-padding)/16), floor((max(a[0],b[0])+padding)/16)+1):
            for y in range(floor((min(a[1],b[1])-padding)/16), floor((max(a[1],b[1])+padding)/16)+1):
                yield (x,y)
    def add(a,b,owner,cat):
        idx=len(accepted)
        accepted.append({'a':a,'b':b,'owner':owner,'category':cat,'labels':[]})
        for cell in cells(a,b): grid[cell].append(idx)
    for owner, crack in enumerate(active):
        cat=crack['category'] if crack['category'] in CATEGORIES else 'unclassified'
        for a,b in zip(crack['geometry'],crack['geometry'][1:]):
            dx,dy=b[0]-a[0],b[1]-a[1]
            length=hypot(dx,dy)
            if length < 1e-9: continue
            valid_owners.add(owner); raw+=length
            n=max(1,ceil(length/16))
            for piece in range(n):
                p=(a[0]+dx*piece/n,a[1]+dy*piece/n)
                q=(a[0]+dx*(piece+1)/n,a[1]+dy*(piece+1)/n)
                vx,vy=q[0]-p[0],q[1]-p[1]
                size=hypot(vx,vy)
                candidates=set()
                for cell in cells(p,q,tolerance_px+1e-8): candidates.update(grid.get(cell,()))
                covered=[]
                for idx in sorted(candidates):
                    old=accepted[idx]; oa,ob=old['a'],old['b']
                    ox,oy=ob[0]-oa[0],ob[1]-oa[1]; olen=hypot(ox,oy)
                    ux,uy=ox/olen,oy/olen
                    longitudinal=vx*ux+vy*uy
                    if abs(longitudinal)/size < (1-1e-10 if tolerance_px==0 else .996194698): continue
                    offset=(p[0]-oa[0])*ux+(p[1]-oa[1])*uy
                    lo,hi=sorted((-offset/longitudinal,(olen-offset)/longitudinal))
                    lo,hi=max(0.,lo),min(1.,hi)
                    cross=(p[0]-oa[0])*uy-(p[1]-oa[1])*ux
                    slope=vx*uy-vy*ux
                    tol=tolerance_px+1e-8
                    if abs(slope)<1e-10:
                        if abs(cross)>tol: continue
                    else:
                        r,s=sorted(((-tol-cross)/slope,(tol-cross)/slope))
                        lo,hi=max(lo,r),min(hi,s)
                    if (hi-lo)*size < 1e-7: continue
                    covered.append((lo,hi)); join(owner,old['owner'])
                    if cat != old['category']:
                        first,last=sorted(((offset+lo*longitudinal)/olen,(offset+hi*longitudinal)/olen))
                        old['labels'].append((max(0,first),min(1,last),cat))
                cursor=0.
                for lo,hi in sorted(covered):
                    if lo>cursor+1e-9:
                        add((p[0]+vx*cursor,p[1]+vy*cursor),(p[0]+vx*lo,p[1]+vy*lo),owner,cat)
                    cursor=max(cursor,hi)
                if cursor<1-1e-9:
                    add((p[0]+vx*cursor,p[1]+vy*cursor),q,owner,cat)
    lengths=dict.fromkeys(CATEGORIES,0.)
    orientation=[0.]*18
    segments=[]
    for item in accepted:
        a,b=item['a'],item['b']; dx,dy=b[0]-a[0],b[1]-a[1]; length=hypot(dx,dy)
        events=defaultdict(int)
        for lo,hi,_ in item['labels']: events[lo]+=1;events[hi]-=1
        events[0.]+=0;events[1.]+=0
        previous=0.; conflicts=0
        for position in sorted(events):
            if position>previous:
                cat='conflict' if conflicts else item['category']
                value=(position-previous)*length
                lengths[cat]+=value
                orientation[min(17,int((degrees(atan2(dy,dx))%180)//10))]+=value
                segments.append({'category':cat,'length_px':value,'geometry':[
                    [a[0]+dx*previous,a[1]+dy*previous],[a[0]+dx*position,a[1]+dy*position]]})
            conflicts+=events[position];previous=position
    from .length_analysis import SEALED
    sealed=sum(lengths[c] for c in SEALED)
    total=sum(lengths.values()); denominator=lengths['open']+sealed
    groups=defaultdict(list)
    for i in sorted(valid_owners): groups[root(i)].append(active[i].get('id',i))
    return {'unit':'px','length_px':lengths,'total_px':total,'raw_length_px':raw,
            'overlap_removed_px':max(0,raw-total),'raw_trace_count':len(valid_owners),
            'trace_count':len(groups),'overlap_groups':[ids for ids in groups.values() if len(ids)>1],
            'tolerance_px':tolerance_px,'filled_fraction':sealed/denominator if denominator else None,
            'orientation_length_px':orientation,'segments':segments,
            'method':f'Union des portions parallèles superposées, tolérance {tolerance_px:g} px ; contacts ponctuels exclus.'}
