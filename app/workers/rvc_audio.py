from pathlib import Path

def trim_edges(source, scratch):
    """Exclude long silent edges from inference without shifting the song."""
    import wave
    try:
        with wave.open(str(source),'rb') as f:
            if f.getnframes()/f.getframerate()<4:
                return Path(source), None
    except (wave.Error, EOFError):
        pass
    import numpy as np
    import soundfile as sf
    x,sr=sf.read(source,always_2d=True)
    hop=max(1,round(sr*.02))
    padded=np.pad(x,((0,(-len(x))%hop),(0,0)))
    energy=np.sqrt(np.mean(padded.reshape(-1,hop,x.shape[1])**2,axis=1)).max(axis=1)
    active=np.flatnonzero(energy>max(1e-5,float(energy.max())*.003))
    if not len(active):
        return Path(source),None
    first=int(active[0])*hop;last=min(len(x),(int(active[-1])+1)*hop)
    start=max(0,first-round(.5*sr)) if first>2*sr else 0
    end=min(len(x),last+round(.5*sr)) if len(x)-last>2*sr else len(x)
    if start==0 and end==len(x):return Path(source),None
    dest=Path(scratch)/'rvc_active.wav';sf.write(dest,x[start:end],sr,subtype='FLOAT')
    return dest,(start/sr,len(x)/sr)

def restore_edges(output, timing):
    if timing is None:return
    import numpy as np
    import soundfile as sf
    x,sr=sf.read(output,always_2d=True)
    start=round(timing[0]*sr);total=round(timing[1]*sr)
    result=np.zeros((total,x.shape[1]),dtype=x.dtype)
    count=min(len(x),total-start);result[start:start+count]=x[:count]
    sf.write(output,result,sr,subtype='PCM_16')
