
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Mine Subsidence Simulator", layout="wide")
st.title("Mine Subsidence Monitoring Simulator")
st.caption("Smooth two-node deformation, independent tilt, vibration, crack-proxy and risk simulation")

with st.sidebar:
    st.header("Simulation Controls")
    duration = st.slider("Duration (s)", 30, 300, 120, 5)
    roof_start = st.slider("Initial roof distance (cm)", 35.0, 60.0, 50.0, 0.5)
    movement = st.slider("Subsidence rate (cm/s)", 0.00, 0.50, 0.10, 0.01)
    st.subheader("Node 1 tilt")
    t1rate = st.slider("Growth (°/s)", 0.00, 0.25, 0.04, 0.01)
    t1bias = st.slider("Initial tilt (°)", -10.0, 10.0, 0.0, 0.5)
    st.subheader("Node 2 tilt")
    t2rate = st.slider("Growth (°/s)", 0.00, 0.25, 0.06, 0.01)
    t2bias = st.slider("Initial tilt (°)", -10.0, 10.0, 0.0, 0.5)
    st.subheader("Other sensors")
    vibbase = st.slider("Base vibration (g proxy)", 0.00, 1.00, 0.15, 0.01)
    crack_onset = st.slider("Crack-proxy onset distance (cm)", 10.0, 30.0, 22.0, 0.5)
    noise = st.slider("Sensor noise", 0.00, 0.30, 0.03, 0.01)
    st.subheader("Display")
    show_animation = st.checkbox("Animate roof + nodes", True)
    playback = st.slider("Playback speed", 0.25, 4.0, 1.0, 0.25)

dt = 0.2
t = np.arange(0, duration + dt, dt)
rng = np.random.default_rng(42)

# Smooth deformation curve
drop = movement*t + 0.025*np.maximum(0, 30-(roof_start-movement*t))*t
roof = np.maximum(5, roof_start-drop)
rate = np.gradient(roof, t)
acc = np.abs(np.gradient(rate, t))

# Independent node measurements
d1 = roof + 0.35*np.sin(t/7) + rng.normal(0, noise, len(t))
d2 = roof - 0.35*np.sin(t/8+0.5) + rng.normal(0, noise, len(t))

tilt1 = t1bias + t1rate*t + 0.8*np.sin(t/12) + rng.normal(0, noise*0.3, len(t))
tilt2 = t2bias + t2rate*t + 1.0*np.sin(t/14+0.8) + rng.normal(0, noise*0.3, len(t))

v1 = np.clip(vibbase + 0.9*acc + .015*np.maximum(0,30-d1)
             + rng.normal(0,noise*.3,len(t)),0,None)
v2 = np.clip(vibbase + 0.8*acc + .017*np.maximum(0,30-d2)
             + rng.normal(0,noise*.3,len(t)),0,None)
temp = 26.5 + .012*t + .25*np.sin(t/20) + rng.normal(0,noise*.2,len(t))

cr1 = np.clip((crack_onset-d1)/max(crack_onset-10,1)*100,0,100)
cr2 = np.clip((crack_onset-d2)/max(crack_onset-10,1)*100,0,100)

def risk(d,tilt,vib,cr):
    ds=np.clip((30-d)/10*100,0,100)
    ts=np.clip(np.abs(tilt)/15*100,0,100)
    vs=np.clip(vib/1.5*100,0,100)
    return np.clip(.40*ds+.30*ts+.15*vs+.15*cr,0,100)

r1,r2=risk(d1,tilt1,v1,cr1),risk(d2,tilt2,v2,cr2)
nr=np.maximum(r1,r2)
state="CRITICAL" if nr[-1]>=75 else "WARNING" if nr[-1]>=40 else "SAFE"

def ttc(d):
    n=max(20,min(150,len(d)//5))
    slope=np.polyfit(t[-n:],d[-n:],1)[0]
    return np.inf if slope>=-1e-6 else max(0,(d[-1]-20)/abs(slope))

ttc1,ttc2=ttc(d1),ttc(d2)

a,b,c,d,e=st.columns(5)
a.metric("Network",state)
b.metric("Node 1 distance",f"{d1[-1]:.1f} cm")
c.metric("Node 2 distance",f"{d2[-1]:.1f} cm")
d.metric("Node 1 tilt",f"{tilt1[-1]:.1f}°")
e.metric("Node 2 tilt",f"{tilt2[-1]:.1f}°")

a,b,c=st.columns(3)
a.metric("Node 1 TTC","—" if not np.isfinite(ttc1) else f"{ttc1:.1f} s")
b.metric("Node 2 TTC","—" if not np.isfinite(ttc2) else f"{ttc2:.1f} s")
c.metric("Network risk",f"{nr[-1]:.0f}/100")

st.subheader("Two-node mine cross-section")
fig=go.Figure()
fig.add_trace(go.Scatter(x=[0,10],y=[roof[-1]]*2,mode="lines",line=dict(width=18),name="Movable roof"))
fig.add_trace(go.Scatter(x=[0,10],y=[0,0],mode="lines",line=dict(width=10),name="Floor"))

def segment(x,y,ang,L=3):
    q=np.deg2rad(ang)
    return [x-L*np.sin(q)/2,x+L*np.sin(q)/2],[y-L*np.cos(q)/2,y+L*np.cos(q)/2]

x,y=segment(3.2,10,tilt1[-1])
fig.add_trace(go.Scatter(x=x,y=y,mode="lines+markers",line=dict(width=12),marker=dict(size=12),name=f"Node 1 ({tilt1[-1]:.1f}°)"))
x,y=segment(6.8,10,tilt2[-1])
fig.add_trace(go.Scatter(x=x,y=y,mode="lines+markers",line=dict(width=12),marker=dict(size=12),name=f"Node 2 ({tilt2[-1]:.1f}°)"))

for x0,dist,label in [(3.2,d1[-1],"N1"),(6.8,d2[-1],"N2")]:
    fig.add_trace(go.Scatter(x=[x0,x0],y=[0,dist],mode="lines",line=dict(dash="dot"),name=f"{label} distance"))
    fig.add_annotation(x=x0+.35,y=dist/2,text=f"{label}: {dist:.1f} cm",showarrow=False)

fig.update_layout(height=430,margin=dict(l=20,r=20,t=20,b=20),
                  xaxis=dict(range=[0,10],visible=False),
                  yaxis=dict(range=[0,max(55,roof_start+5)],title="Height / distance (cm)"))
st.plotly_chart(fig,use_container_width=True)

st.subheader("Telemetry")
tabs=st.tabs(["Subsidence","Tilt","Vibration + Temperature","Crack proxy + Risk"])

with tabs[0]:
    f=go.Figure()
    f.add_trace(go.Scatter(x=t,y=d1,mode="lines",name="Node 1"))
    f.add_trace(go.Scatter(x=t,y=d2,mode="lines",name="Node 2"))
    f.add_hline(y=30,line_dash="dash",annotation_text="30 cm warning")
    f.add_hline(y=20,line_dash="dash",annotation_text="20 cm critical")
    f.update_layout(height=420,xaxis_title="Time (s)",yaxis_title="Distance (cm)")
    st.plotly_chart(f,use_container_width=True)

with tabs[1]:
    f=go.Figure()
    f.add_trace(go.Scatter(x=t,y=tilt1,mode="lines",name="Node 1"))
    f.add_trace(go.Scatter(x=t,y=tilt2,mode="lines",name="Node 2"))
    f.add_hline(y=15,line_dash="dash",annotation_text="+15°")
    f.add_hline(y=-15,line_dash="dash",annotation_text="-15°")
    f.update_layout(height=420,xaxis_title="Time (s)",yaxis_title="Tilt (°)")
    st.plotly_chart(f,use_container_width=True)

with tabs[2]:
    f=go.Figure()
    f.add_trace(go.Scatter(x=t,y=v1,mode="lines",name="Node 1 vibration"))
    f.add_trace(go.Scatter(x=t,y=v2,mode="lines",name="Node 2 vibration"))
    f.add_trace(go.Scatter(x=t,y=temp,mode="lines",name="Temperature",yaxis="y2"))
    f.update_layout(height=420,xaxis_title="Time (s)",yaxis=dict(title="Vibration"),
                    yaxis2=dict(title="Temperature °C",overlaying="y",side="right"))
    st.plotly_chart(f,use_container_width=True)

with tabs[3]:
    f=go.Figure()
    f.add_trace(go.Scatter(x=t,y=cr1,mode="lines",name="Node 1 crack proxy"))
    f.add_trace(go.Scatter(x=t,y=cr2,mode="lines",name="Node 2 crack proxy"))
    f.add_trace(go.Scatter(x=t,y=r1,mode="lines",name="Node 1 risk",line=dict(dash="dot")))
    f.add_trace(go.Scatter(x=t,y=r2,mode="lines",name="Node 2 risk",line=dict(dash="dot")))
    f.add_hline(y=40,line_dash="dash",annotation_text="Warning")
    f.add_hline(y=75,line_dash="dash",annotation_text="Critical")
    f.update_layout(height=420,xaxis_title="Time (s)",yaxis_title="Score (%)")
    st.plotly_chart(f,use_container_width=True)

df=pd.DataFrame({
"time_s":t,"node1_distance_cm":d1,"node2_distance_cm":d2,
"node1_tilt_deg":tilt1,"node2_tilt_deg":tilt2,
"node1_vibration":v1,"node2_vibration":v2,"temperature_C":temp,
"node1_crack_proxy_pct":cr1,"node2_crack_proxy_pct":cr2,
"node1_risk":r1,"node2_risk":r2,"network_risk":nr
})
st.subheader("Generated dataset")
st.dataframe(df.round(3),use_container_width=True,height=320)
st.download_button("Download CSV",df.to_csv(index=False).encode(),"mine_subsidence_simulation.csv","text/csv",use_container_width=True)
st.info("Crack is a deformation-derived proxy, not direct crack width. TTC is estimated time-to-critical deformation, not a validated collapse prediction. Tilt is independently visualized for Node 1 and Node 2.")
