"""Export a recorded replay as a minimap MP4, PNG, and self-contained player.

python -m lanerl_jax.replay_render trace.npz --out-dir output --video
Requires numpy/Pillow and ffmpeg; no JAX, GPU, server, or real client.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BUTTONS = ('noop', 'move', 'attack_move', 'Q', 'W', 'E', 'R', 'recall')
COLORS = ('#59b8ff', '#ff787e')


def load(path):
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files if k != 'metadata'}
        meta = json.loads(str(z['metadata']))
    return d, meta


def map_image(d):
    return Image.fromarray(np.where(d['walkable'][::-1, :, None],
                                   np.array([53, 70, 62], np.uint8),
                                   np.array([22, 31, 36], np.uint8)))


def unit_name(d, i, u):
    if u < 0 or u >= d['kind'].shape[1]:
        return 'none'
    side = ('blue', 'red', 'neutral')[int(d['team'][i, u])]
    kind = ('empty', 'Garen', 'minion', 'turret')[int(d['kind'][i, u])]
    return f'{side} {kind} #{u}'


def action_name(d, i, side):
    if 'order_resolved' in d and not d['order_resolved'][i, side]:
        name = BUTTONS[int(d['button'][i, side])]
        return name + (' click' if name in ('move', 'attack_move', 'r') else '')
    kind = int(d['order_kind'][i, side])
    if kind == 2:
        return 'attack ' + unit_name(d, i, int(d['order_target'][i, side]))
    if kind == 1:
        return 'move'
    if kind == 0 and d['button'][i, side] == 1:
        return 'move suppressed (minimap)'
    return BUTTONS[int(d['button'][i, side])]


def render_frame(d, meta, i, background, view="map"):
    if view == "combat":
        return render_combat(d, meta, i, background)
    im = Image.new('RGB', (1280, 800), '#10181f')
    draw = ImageDraw.Draw(im)
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = ImageFont.truetype(font_path, 17)
    small = ImageFont.truetype(font_path, 14)
    title = ImageFont.truetype(font_path, 22)
    t = float(d['t_ms'][i]) / 1000
    draw.text((24, 12), f"{meta['label']}    {int(t//60):02d}:{t%60:04.1f}", font=title, fill='#f0f4f6')
    ter = meta['terrain']
    x0, y0, cs = ter['min_x'], ter['min_y'], ter['cell_size']
    h, w = d['walkable'].shape

    def panel(rect, bounds, zoom=False):
        left, top, size = rect
        bx, by, bw, bh = bounds
        crop = ((bx-x0)/cs, h-(by+bh-y0)/cs, (bx+bw-x0)/cs, h-(by-y0)/cs)
        tile = background.crop(crop).resize((size, size), Image.Resampling.NEAREST)
        tile_draw = ImageDraw.Draw(tile)
        def point(x, y):
            return ((float(x)-bx)/bw*size, size-(float(y)-by)/bh*size)
        for side in range(2):
            if not d['alive'][i, side]:
                continue
            p = point(d['x'][i, side], d['y'][i, side])
            k, target = int(d['order_kind'][i, side]), int(d['order_target'][i, side])
            if k == 1:
                q = point(d.get('click_x', d['order_x'])[i, side],
                          d.get('click_y', d['order_y'])[i, side])
                color = '#dba7ff'
            elif k == 2 and target >= 0 and d['alive'][i, target]:
                q = (point(d['click_x'][i, side], d['click_y'][i, side])
                     if 'click_x' in d else point(d['x'][i, target], d['y'][i, target]))
                color = '#ffe185'
            else:
                continue
            tile_draw.line([p, q], fill=color, width=2)
            tile_draw.ellipse((q[0]-4, q[1]-4, q[0]+4, q[1]+4), outline=color, width=2)
        for u in np.flatnonzero(d['kind'][i] > 0):
            alive = bool(d['alive'][i, u])
            if not alive and u >= 2:
                continue
            x, y = point(d['x'][i, u], d['y'][i, u])
            if not (-20 < x < size+20 and -20 < y < size+20):
                continue
            k = int(d['kind'][i, u]); side = int(d['team'][i, u])
            color = COLORS[side] if alive else '#7e858c'
            r = (10 if zoom else 8) if k == 1 else ((6 if zoom else 3) if k == 2 else 7)
            if k == 1:
                tile_draw.ellipse((x-r,y-r,x+r,y+r), fill=color, outline='#ffffff', width=2)
                tile_draw.text((x+12, y-10), ('B','R')[side] + ('' if alive else ' dead'), font=small, fill=color)
                if d['e_active'][i, side]:
                    er = 330/bw*size
                    tile_draw.ellipse((x-er,y-er,x+er,y+er), outline=color, width=2)
            elif k == 3:
                tile_draw.polygon([(x,y-r),(x+r,y),(x,y+r),(x-r,y)], fill=color)
            else:
                tile_draw.rectangle((x-r,y-r,x+r,y+r), fill=color)
            if zoom or k == 1:
                frac = float(d['hp'][i,u]/max(d['max_hp'][i,u], 1))
                tile_draw.rectangle((x-10,y-r-7,x+10,y-r-4), fill='#121212')
                tile_draw.rectangle((x-10,y-r-7,x-10+20*max(0,min(1,frac)),y-r-4), fill=color)
        im.paste(tile, (left, top))

    panel((24, 56, 704), (x0, y0, w*cs, h*cs))
    cx, cy = d['x'][i, 0], d['y'][i, 0]
    panel((768, 56, 432), (cx-1600, cy-1600, 3200, 3200), True)
    draw.text((778, 64), 'BLUE GAREN CLOSE-UP', font=small, fill='#f0f4f6')
    for side in range(2):
        y = 510 + side*105
        draw.text((768,y), f"{('BLUE','RED')[side]}   CS {int(d['cs'][i,side])}   deaths {int(d['deaths'][i,side])}   lvl {int(d['level'][i,side])}", font=font, fill=COLORS[side])
        draw.text((768,y+27), action_name(d,i,side), font=small, fill='#f0f4f6')
        draw.text((768,y+49), f"windup {d['aa_windup'][i,side]:.2f}s   cooldown {d['aa_cooldown'][i,side]:.2f}s", font=small, fill='#b9c4cd')
        held = int(d['target'][i,side])
        draw.text((768,y+70), 'held target: '+unit_name(d,i,held), font=small, fill='#b9c4cd')
    draw.text((24,771), 'Circles: Garen  |  squares: minions  |  diamonds: turrets  |  purple: move  |  yellow: attack', font=small, fill='#c4cdd5')
    draw.text((768,746), 'All units visible; policy still uses its own fog.', font=small, fill='#c4cdd5')
    return im




def camera_offsets(meta):
    """Canonical camera footprint, including the excluded minimap notch."""
    if 'camera_frame' not in meta:
        return None
    from lanerl_rl.projection import screen_to_world_centred, MINIMAP_X_MIN, MINIMAP_Y_MIN
    axis=np.asarray(meta['camera_frame']['axis']); normal=np.asarray(meta['camera_frame']['normal'])
    perimeter=[(0,0),(1,0),(1,MINIMAP_Y_MIN),(MINIMAP_X_MIN,MINIMAP_Y_MIN),(MINIMAP_X_MIN,1),(0,1)]
    offsets=[]
    for side in (1,-1):
        offsets.append([(side*ds*axis+dn*normal).tolist()
                        for ds,dn in (screen_to_world_centred(0,0,sx,sy) for sx,sy in perimeter)])
    return offsets

def combat_state(d, i):
    """Compact diagnostic state; absent instrumentation stays explicitly absent."""
    units = []
    for u in np.flatnonzero((d['alive'][i] & (d['kind'][i] > 0)) | (np.arange(d['x'].shape[1]) < 2)):
        units.append([int(u), float(d['hp'][i,u]), float(d['max_hp'][i,u]),
                      int(d['target'][i,u]), int(d['aa_target'][i,u]),
                      float(d['aa_windup'][i,u]), bool(d['is_attacking'][i,u]),
                      int(d['spawn_seq'][i,u]), int(d['model'][i,u]) if 'model' in d else -1])
    missiles = None
    if 'missile_alive' in d:
        missiles = []
        for m in np.flatnonzero(d['missile_alive'][i]):
            source = int(d['missile_source'][i,m])
            identity = (0 <= source < d['x'].shape[1] and
                        ('missile_source_seq' not in d or d['missile_source_seq'][i,m] < 0 or
                         d['spawn_seq'][i,source] == d['missile_source_seq'][i,m]))
            missiles.append([int(m), float(d['missile_x'][i,m]), float(d['missile_y'][i,m]),
                             int(d['missile_tx'][i,m]), source,
                             int(d['team'][i,source]) if identity else -1,
                             float(d['missile_damage'][i,m])])
    champs = []
    for u in range(2):
        route = []
        if 'waypoints' in d:
            start, end = int(d['waypoint_key'][i,u]), int(d['n_waypoints'][i,u])
            route = d['waypoints'][i,u,max(0,start):end].astype(float).tolist()
        # Net health loss is not damage attribution: healing and damage can overlap.
        old = max(0, int(np.searchsorted(d['t_ms'], d['t_ms'][i]-1000)))
        same_life = (d['spawn_seq'][old,u] == d['spawn_seq'][i,u] and
                     d['deaths'][old,u] == d['deaths'][i,u] and
                     bool(np.all(d['alive'][old:i+1,u])))
        net = float(d['hp'][i,u]-d['hp'][old,u]) if same_life else None
        champs.append(dict(route=route, route_status=int(d['route_status'][i,u]) if 'route_status' in d else None,
                           net_hp_1s=net,
                           active=[bool(d[k][i,u]) if k in d else None for k in ('q_active','w_active','e_active')],
                           r_cast_ms=float(d['r_cast_ms'][i,u]) if 'r_cast_ms' in d else None,
                           ranks=d['spell_level'][i,u].astype(int).tolist() if 'spell_level' in d else None,
                           cooldown=d['spell_cooldown'][i,u].astype(float).tolist() if 'spell_cooldown' in d else None))
    return dict(units=units, missiles=missiles, champs=champs)


def wave_center(d, i):
    # Separate waves may be thousands of units apart: their global median
    # points at empty terrain. Follow an engaged cluster, blue priority.
    mask = d['alive'][i] & (d['kind'][i] == 2) & (d['x'][i] < 6500) & (d['y'][i] > 8000)
    if not np.any(mask):
        return 1900., 12100.
    points=np.stack((d['x'][i,mask],d['y'][i,mask]),axis=1)
    for side in range(2):
        anchor=np.array([d['x'][i,side],d['y'][i,side]])
        nearby=np.linalg.norm(points-anchor,axis=1)<1100
        if d['alive'][i,side] and np.any(nearby):
            return tuple(np.median(np.vstack((points[nearby],anchor)),axis=0))
    # No engaged champion: the densest local minion group is the useful view.
    neighbors=np.linalg.norm(points[:,None]-points[None,:],axis=2)<1100
    cluster=neighbors[np.argmax(neighbors.sum(1))]
    return tuple(np.median(points[cluster],axis=0))


def render_combat(d, meta, i, background):
    """A deliberately simple game view drawn only from recorded simulator state."""
    im = Image.new('RGB', (1280,800), '#10181f'); draw = ImageDraw.Draw(im)
    font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font=ImageFont.truetype(font_path,16); small=ImageFont.truetype(font_path,12)
    info=combat_state(d,i); extra={u[0]:u for u in info['units']}
    t=float(d['t_ms'][i])/1000
    origin = 'SOURCE SERVER' if meta.get('environment') == 'source-server' else 'SIMULATOR'
    draw.text((20,12), f"{meta['label']}   {int(t//60):02d}:{t%60:04.1f}   {origin} COMBAT VIEW",font=font,fill='white')
    ter=meta['terrain']; x0,y0,cs=ter['min_x'],ter['min_y'],ter['cell_size']; h=background.height
    def panel(left,top,size,bx,by,bw,bh,detail):
        tile=background.crop(((bx-x0)/cs,h-(by+bh-y0)/cs,(bx+bw-x0)/cs,h-(by-y0)/cs)).resize((size,size),Image.Resampling.NEAREST)
        dr=ImageDraw.Draw(tile)
        def point(x,y): return ((float(x)-bx)/bw*size,size-(float(y)-by)/bh*size)
        def line(a,b,color,width=1): dr.line([a,b],fill=color,width=width)
        for u,e in extra.items():
            if not d['alive'][i,u]: continue
            a=point(d['x'][i,u],d['y'][i,u]); target=e[4] if e[6] and e[5]>0 else e[3]
            if detail and 0<=target<d['x'].shape[1] and d['alive'][i,target]:
                b=point(d['x'][i,target],d['y'][i,target]); winding=e[6] and e[5]>0
                line(a,b,'#fff29a' if winding else '#64747d',3 if winding else 1)
                if winding: dr.ellipse((b[0]-8,b[1]-8,b[0]+8,b[1]+8),outline='#fff29a',width=2)
        for u in range(2):
            a=point(d['x'][i,u],d['y'][i,u]); route=info['champs'][u]['route']
            if route and d['alive'][i,u]:
                dr.line([a]+[point(*p) for p in route],fill='#64efcb',width=2)
            if detail and d['alive'][i,u] and int(d['order_kind'][i,u]) in (1,2):
                target=int(d['order_target'][i,u]); k=int(d['order_kind'][i,u])
                if 'click_x' in d: b=point(d['click_x'][i,u],d['click_y'][i,u])
                elif k==2 and target>=0: b=point(d['x'][i,target],d['y'][i,target])
                else: b=point(d['order_x'][i,u],d['order_y'][i,u])
                dr.ellipse((b[0]-5,b[1]-5,b[0]+5,b[1]+5),outline='#dba7ff',width=2)
        for m in info['missiles'] or []:
            _,mx,my,target,source,team,damage=m; a=point(mx,my); color=COLORS[team] if team>=0 else '#ffffff'
            if 0<=target<d['x'].shape[1]:
                b=point(d['x'][i,target],d['y'][i,target]); dx,dy=b[0]-a[0],b[1]-a[1]; length=max(1,(dx*dx+dy*dy)**.5)
                line((a[0]-dx/length*12,a[1]-dy/length*12),a,color,3)
            dr.ellipse((a[0]-3,a[1]-3,a[0]+3,a[1]+3),fill='white')
        for u,e in extra.items():
            alive=bool(d['alive'][i,u]); k=int(d['kind'][i,u]); side=int(d['team'][i,u]); color=COLORS[side] if alive else '#777777'
            x,y=point(d['x'][i,u],d['y'][i,u]); r=11 if k==1 else 5 if k==2 else 9
            if not (-30<x<size+30 and -30<y<size+30): continue
            if k==1:
                dr.ellipse((x-r,y-r,x+r,y+r),fill=color,outline='white',width=2)
                if info['champs'][u]['active'][2]:
                    er=330/bw*size; dr.ellipse((x-er,y-er,x+er,y+er),outline=color,width=2)
            elif k==3: dr.polygon([(x,y-r),(x+r,y),(x,y+r),(x-r,y)],fill=color)
            else: dr.rectangle((x-r,y-r,x+r,y+r),fill=color)
            if detail:
                frac=max(0,min(1,e[1]/max(e[2],1))); dr.rectangle((x-16,y-r-8,x+16,y-r-4),fill='#111111');dr.rectangle((x-16,y-r-8,x-16+32*frac,y-r-4),fill=color)
                profile_label=meta.get('profile_label',[p['label'] for p in meta.get('profiles',[])])
                label=profile_label[e[8]] if 0<=e[8]<len(profile_label) else 'minion'
                short={'melee':'M','caster':'C','cannon':'S','super':'U'}.get(label,'m')
                name=('B Garen' if side==0 else 'R Garen') if k==1 else ('T' if k==3 else short)+str(u)
                dr.text((x+9,y+5),f'{name} {e[1]:.0f}' if k==1 else name,font=small,fill=color)
                if k==1 and e[6] and e[5]>0: dr.text((x+9,y-23),f'AA >{e[4]} {e[5]:.2f}s',font=small,fill='#fff29a')
        im.paste(tile,(left,top))
    cx,cy=wave_center(d,i);panel(20,50,710,cx-900,cy-900,1800,1800,True)
    panel(760,50,330,-500,7500,7500,7500,False)
    draw.text((767,58),'TOP LANE OVERVIEW',font=small,fill='white')
    for side in range(2):
        y=395+side*142; c=info['champs'][side]; color=COLORS[side]
        draw.text((760,y),f"{('BLUE','RED')[side]} Garen   HP {d['hp'][i,side]:.0f}/{d['max_hp'][i,side]:.0f}   CS {d['cs'][i,side]}",font=font,fill=color)
        draw.text((760,y+24),action_name(d,i,side),font=small,fill='white')
        draw.text((760,y+43),'held: '+unit_name(d,i,int(d['target'][i,side])),font=small,fill='#c5d0d6')
        active=' '.join(k+(' locked' if c['ranks'] is not None and c['ranks'][j]==0 else ' ON' if v else ' off' if v is not None else ' ?') for j,(k,v) in enumerate(zip('QWE',c['active'])))
        draw.text((760,y+62),active+('   R CAST' if c['r_cast_ms'] and c['r_cast_ms']>0 else ''),font=small,fill='#c5d0d6')
        cds='unavailable' if c['cooldown'] is None else ' / '.join('-' if c['ranks'] is not None and c['ranks'][j]==0 else f'{v:.1f}' for j,v in enumerate(c['cooldown']))
        draw.text((760,y+81),'Q/W/E/R cooldown s: '+cds,font=small,fill='#c5d0d6')
        net='n/a (life transition)' if c['net_hp_1s'] is None else f"{c['net_hp_1s']:+.0f}"
        draw.text((760,y+100),f"1s net HP: {net}   route status: {c['route_status']}",font=small,fill='#c5d0d6')
    draw.text((760,700),'Projectile positions: '+('recorded' if info['missiles'] is not None else 'UNAVAILABLE in this trace'),font=small,fill='#c5d0d6')
    draw.text((760,723),'Net HP includes healing; not damage attribution.',font=small,fill='#c5d0d6')
    cap=d['missile_alive'].shape[1] if 'missile_alive' in d else '?'
    count=len(info['missiles']) if info['missiles'] is not None else '?'
    draw.text((760,746),f"Alive minions: {int(np.sum(d['alive'][i] & (d['kind'][i]==2)))}   missiles: {count}/{cap}",font=small,fill='#c5d0d6')
    attack_note = ('Swing victim unavailable' if meta.get('melee_victim_recorded') is False
                   else 'Yellow: winding attack > victim')
    draw.text((20,774),attack_note+'   Gray: held target   Mint: remaining route   Purple ring: click   E: radius ring',font=small,fill='#c5d0d6')
    return im

def export_html(d, meta, background, destination, view="map", start_seconds=0):
    # Ten snapshots per game second; full-resolution decisions remain in NPZ.
    indices = np.unique(np.searchsorted(d['t_ms'], np.arange(d['t_ms'][0],d['t_ms'][-1],100))).tolist()
    indices.append(len(d['t_ms'])-1)
    frames = []
    for i in indices:
        units = []
        for u in np.flatnonzero((d['alive'][i] & (d['kind'][i]>0)) | (np.arange(d['x'].shape[1])<2)):
            units.append([int(u), round(float(d['x'][i,u]),1), round(float(d['y'][i,u]),1),
                          round(float(d['hp'][i,u]/max(d['max_hp'][i,u],1)),3),
                          int(d['kind'][i,u]), int(d['team'][i,u]), int(d['alive'][i,u])])
        champs = [[int(d[k][i,s]) for k in ('cs','deaths','level')]
                  + [action_name(d,i,s), round(float(d['aa_windup'][i,s]),3),
                     round(float(d['aa_cooldown'][i,s]),3), int(d['order_kind'][i,s]),
                     int(d['order_target'][i,s]), float(d.get('click_x', d['order_x'])[i,s]), float(d.get('click_y', d['order_y'])[i,s]),
                     unit_name(d,i,int(d['target'][i,s]))]
                  for s in range(2)]
        frames.append([round(float(d['t_ms'][i])/1000,3),units,champs,combat_state(d,i)])
    buf=io.BytesIO(); background.save(buf,format='PNG')
    payload=dict(view=view, start_seconds=start_seconds, sample_hz=10, missile_capacity=int(d['missile_alive'].shape[1]) if 'missile_alive' in d else None, camera_offsets=camera_offsets(meta), cursor_coordinates='click_x' in d, meta=meta, frames=frames, height=background.height,width=background.width,
                 map='data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode())
    template=Path(__file__).with_name('replay_player.html').read_text()
    destination.write_text(template.replace('__REPLAY_DATA__', json.dumps(payload,separators=(',',':')).replace('</','<\\/')))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('trace',type=Path)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--video',action='store_true')
    p.add_argument('--speed',type=float,default=10)
    p.add_argument('--view',choices=('map','combat'),default='map')
    p.add_argument('--start-seconds',type=float,default=0)
    p.add_argument('--end-seconds',type=float)
    p.add_argument('--fps',type=int,default=15)
    args=p.parse_args()
    if args.speed<=0 or args.fps<=0: p.error('speed and fps must be positive')
    d,meta=load(args.trace); bg=map_image(d)
    start=max(float(d['t_ms'][0]),args.start_seconds*1000)
    end=min(float(d['t_ms'][-1]),args.end_seconds*1000 if args.end_seconds is not None else float(d['t_ms'][-1]))
    if not np.isfinite(start+end+args.speed) or start>=end: p.error('video interval must be finite, nonempty, and overlap trace')
    args.out_dir.mkdir(parents=True,exist_ok=True)
    export_html(d,meta,bg,args.out_dir/'replay.html',args.view,args.start_seconds)
    preview=int(np.searchsorted(d['t_ms'],(start+end)/2 if args.view=='combat' else min(max(180000,start),end)))
    render_frame(d,meta,preview,bg,args.view).save(args.out_dir/'preview.png')
    if args.video:
        cmd=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
             '-s','1280x800','-r',str(args.fps),'-i','-','-an','-c:v','libx264',
             '-preset','veryfast','-crf','23','-pix_fmt','yuv420p','-threads','1',
             '-movflags','+faststart',str(args.out_dir/'replay.mp4')]
        with subprocess.Popen(cmd,stdin=subprocess.PIPE) as proc:
            try:
                for t in np.arange(start,end+0.01,1000*args.speed/args.fps):
                    i=min(int(np.searchsorted(d['t_ms'],t)),len(d['t_ms'])-1)
                    proc.stdin.write(render_frame(d,meta,i,bg,args.view).tobytes())
                proc.stdin.close()
                if proc.wait(): raise RuntimeError('ffmpeg failed')
            except BaseException:
                proc.kill(); raise
    print(args.out_dir)


if __name__=='__main__':
    main()
