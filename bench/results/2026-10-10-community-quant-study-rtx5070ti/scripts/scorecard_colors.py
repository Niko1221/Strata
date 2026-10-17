"""Presentation-only, column-relative heatmap colors; never an aggregate score."""
import math

PALETTE = ('#b35806','#f6c67a','#fff8dc','#80cdc1','#01665e')
MISSING = '#d9dfe5'

def luminance(color):
    rgb=[int(color[i:i+2],16)/255 for i in (1,3,5)]
    linear=[v/12.92 if v<=0.04045 else ((v+0.055)/1.055)**2.4 for v in rgb]
    return sum(v*w for v,w in zip(linear,(0.2126,0.7152,0.0722)))

def contrast(a,b):
    x,y=sorted((luminance(a),luminance(b)))
    return (y+0.05)/(x+0.05)

def cell_style(value,low,high,higher_is_better=True):
    if value is None:
        background=MISSING
    else:
        if not all(math.isfinite(v) for v in (value,low,high)) or high<low:
            raise ValueError('finite values and ordered column bounds required')
        position=0.5 if high==low else max(0,min(1,(value-low)/(high-low)))
        if not higher_is_better:
            position=1-position
        background=PALETTE[min(4,int(position*5))]
    foreground=max(('#ffffff','#14242b'),key=lambda c:contrast(background,c))
    return background,foreground
