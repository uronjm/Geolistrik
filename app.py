import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import griddata
import io

st.set_page_config(
    page_title="Model 3D Resistivitas Geolistrik",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 Model 3D Resistivitas Geolistrik")
st.markdown("**Visualisasi 3D Sebaran Gambut dari Data Geolistrik 2D**")

# ─── Colorscale (mirip Surfer: biru→cyan→hijau→kuning→oranye→merah) ──────────
COLORSCALE = [
    [0.00, '#00008B'],
    [0.10, '#0000FF'],
    [0.20, '#0080FF'],
    [0.30, '#00BFFF'],
    [0.40, '#00FFFF'],
    [0.50, '#00FF80'],
    [0.60, '#80FF00'],
    [0.70, '#FFFF00'],
    [0.80, '#FFA500'],
    [0.90, '#FF2000'],
    [1.00, '#8B0000'],
]

# ─── Load Data ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(file):
    xls = pd.ExcelFile(file)
    geom = pd.read_excel(xls, sheet_name='GEOMETRY')
    data = pd.read_excel(xls, sheet_name='DATA')
    return geom, data

@st.cache_data
def build_grid(data_bytes, nx, ny, nz):
    import pickle, hashlib
    df = pd.read_excel(io.BytesIO(data_bytes), sheet_name='DATA')
    geom = pd.read_excel(io.BytesIO(data_bytes), sheet_name='GEOMETRY')
    line_y = dict(zip(geom['Line'], geom['Y_Position']))
    df['Y'] = df['Line'].map(line_y)

    xi = np.linspace(df['Distance'].min(), df['Distance'].max(), nx)
    yi = np.linspace(df['Y'].min(), df['Y'].max(), ny)
    zi = np.linspace(df['Depth'].min(), df['Depth'].max(), nz)

    points = df[['Distance','Y','Depth']].values
    values = df['Resistivity'].values

    XX, YY, ZZ = np.meshgrid(xi, yi, zi, indexing='ij')
    flat_pts = np.column_stack([XX.flatten(), YY.flatten(), ZZ.flatten()])

    r_lin  = griddata(points, values, flat_pts, method='linear')
    r_near = griddata(points, values, flat_pts, method='nearest')
    r_filled = np.where(np.isnan(r_lin), r_near, r_lin)
    grid_3d = r_filled.reshape(nx, ny, nz)

    return xi, yi, zi, grid_3d, df, geom

st.sidebar.header("📂 Upload Data")
uploaded = st.sidebar.file_uploader("Upload file Excel (.xlsx)", type=['xlsx'])

if uploaded:
    file_bytes = uploaded.read()
    src_label = uploaded.name
else:
    try:
        with open('Geolistrik_3D.xlsx','rb') as f:
            file_bytes = f.read()
        src_label = "Geolistrik_3D.xlsx (bawaan)"
        st.sidebar.success(f"✅ Data: {src_label}")
    except:
        st.warning("⚠️ Silakan upload file Excel data geolistrik Anda.")
        st.stop()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Resolusi Grid")
nx = st.sidebar.slider("Grid X", 20, 100, 60, 10)
ny = st.sidebar.slider("Grid Y", 10, 50, 30, 5)
nz = st.sidebar.slider("Grid Z (Kedalaman)", 10, 40, 20, 5)

st.sidebar.header("📊 Skala Warna")
use_log = st.sidebar.checkbox("Skala Logaritmik", value=True)

xi, yi, zi, grid_3d, df, geom = build_grid(file_bytes, nx, ny, nz)
line_y = dict(zip(geom['Line'], geom['Y_Position']))
df['Y'] = df['Line'].map(line_y)

# Log transform jika dipilih
if use_log:
    plot_grid = np.log10(np.clip(grid_3d, 0.01, None))
    plot_raw  = np.log10(np.clip(df['Resistivity'].values, 0.01, None))
    cbar_title = "Log₁₀ Resistivitas (Ω·m)"
    tickvals = np.log10([0.5, 1, 2, 5, 10, 20, 50])
    ticktext = ['0.5','1','2','5','10','20','50']
else:
    plot_grid = grid_3d.copy()
    plot_raw  = df['Resistivity'].values
    cbar_title = "Resistivitas (Ω·m)"
    tickvals = None
    ticktext = None

cmin = np.nanmin(plot_grid)
cmax = np.nanmax(plot_grid)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🧊 Model 3D Box",
    "🏗️ Fence Diagram",
    "📐 Irisan Horizontal",
    "📏 Irisan Vertikal",
    "📊 Data & Statistik",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — MODEL 3D BOX (seperti Surfer)
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Model 3D Resistivitas — Tampilan Box")

    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        show_front = st.checkbox("Sisi Depan (Y min)", value=True)
        show_back  = st.checkbox("Sisi Belakang (Y max)", value=False)
        show_top   = st.checkbox("Sisi Atas (permukaan)", value=True)
    with col_opt2:
        show_left  = st.checkbox("Sisi Kiri (X min)", value=True)
        show_right = st.checkbox("Sisi Kanan (X max)", value=False)
        show_bot   = st.checkbox("Sisi Bawah (terdalam)", value=False)

    fig = go.Figure()

    colorbar_cfg = dict(
        title=dict(text=cbar_title, side='right'),
        thickness=18, len=0.7,
        tickvals=tickvals, ticktext=ticktext,
    )
    cb_added = False

    def add_surface(fig, x2d, y2d, z2d, color2d, name, show_cb):
        fig.add_trace(go.Surface(
            x=x2d, y=y2d, z=z2d,
            surfacecolor=color2d,
            colorscale=COLORSCALE,
            cmin=cmin, cmax=cmax,
            showscale=show_cb,
            colorbar=colorbar_cfg if show_cb else None,
            name=name,
            hovertemplate='X:%{x:.1f} m  Y:%{y:.1f} m  Z:%{z:.2f} m<extra>' + name + '</extra>',
        ))

    # Sisi Depan — Y=min, xz plane
    if show_front:
        XF, ZF = np.meshgrid(xi, zi, indexing='ij')
        YF = np.full_like(XF, yi[0])
        CF = plot_grid[:, 0, :]
        add_surface(fig, XF, YF, ZF, CF, 'Depan', not cb_added)
        cb_added = True

    # Sisi Belakang — Y=max
    if show_back:
        XB, ZB = np.meshgrid(xi, zi, indexing='ij')
        YB = np.full_like(XB, yi[-1])
        CB = plot_grid[:, -1, :]
        add_surface(fig, XB, YB, ZB, CB, 'Belakang', not cb_added)
        cb_added = True

    # Sisi Atas — Z=max (permukaan)
    if show_top:
        XT, YT = np.meshgrid(xi, yi, indexing='ij')
        ZT = np.full_like(XT, zi[-1])
        CT = plot_grid[:, :, -1]
        add_surface(fig, XT, YT, ZT, CT, 'Atas', not cb_added)
        cb_added = True

    # Sisi Bawah — Z=min
    if show_bot:
        XBo, YBo = np.meshgrid(xi, yi, indexing='ij')
        ZBo = np.full_like(XBo, zi[0])
        CBo = plot_grid[:, :, 0]
        add_surface(fig, XBo, YBo, ZBo, CBo, 'Bawah', not cb_added)
        cb_added = True

    # Sisi Kiri — X=min
    if show_left:
        YL, ZL = np.meshgrid(yi, zi, indexing='ij')
        XL = np.full_like(YL, xi[0])
        CL = plot_grid[0, :, :]
        add_surface(fig, XL, YL, ZL, CL, 'Kiri', not cb_added)
        cb_added = True

    # Sisi Kanan — X=max
    if show_right:
        YR, ZR = np.meshgrid(yi, zi, indexing='ij')
        XR = np.full_like(YR, xi[-1])
        CR = plot_grid[-1, :, :]
        add_surface(fig, XR, YR, ZR, CR, 'Kanan', not cb_added)
        cb_added = True

    # Label lintasan
    for _, row in geom.iterrows():
        fig.add_trace(go.Scatter3d(
            x=[(xi[0]+xi[-1])/2],
            y=[row['Y_Position']],
            z=[zi[-1] + abs(zi[-1]-zi[0])*0.15],
            mode='text+markers',
            text=[f"<b>{row['Line']}<br>Y={row['Y_Position']} m</b>"],
            textfont=dict(size=11, color='black'),
            marker=dict(size=4, color='black'),
            showlegend=False,
        ))
        # Garis vertikal penanda lintasan
        fig.add_trace(go.Scatter3d(
            x=[xi[0], xi[-1]],
            y=[row['Y_Position'], row['Y_Position']],
            z=[zi[-1], zi[-1]],
            mode='lines',
            line=dict(color='black', width=2, dash='dash'),
            showlegend=False,
        ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X (m)', showbackground=True, backgroundcolor='rgb(240,240,240)'),
            yaxis=dict(title='Y (m)', showbackground=True, backgroundcolor='rgb(230,230,240)'),
            zaxis=dict(title='Kedalaman (m)', showbackground=True, backgroundcolor='rgb(230,240,240)'),
            camera=dict(eye=dict(x=1.6, y=-1.8, z=1.1)),
            aspectmode='manual',
            aspectratio=dict(x=2.2, y=0.6, z=0.5),
        ),
        height=620,
        title=dict(text="MODEL 3D RESISTIVITAS", x=0.5, font=dict(size=16, color='black')),
        margin=dict(l=0, r=20, t=50, b=0),
        paper_bgcolor='white',
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info("💡 **Tips:** Centang/uncentang sisi box di atas untuk mengatur tampilan. "
            "Gunakan mouse untuk memutar (drag), zoom (scroll), dan geser (shift+drag).")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — FENCE DIAGRAM
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Fence Diagram — Penampang Vertikal 3 Lintasan")

    fig_fence = go.Figure()
    cb_f = False

    for i, (_, row) in enumerate(geom.iterrows()):
        line_name = row['Line']
        y_pos     = row['Y_Position']

        # Cari index Y terdekat di grid
        iy = np.argmin(np.abs(yi - y_pos))
        face = plot_grid[:, iy, :]   # shape: NX x NZ

        XF2, ZF2 = np.meshgrid(xi, zi, indexing='ij')
        YF2 = np.full_like(XF2, float(y_pos))

        fig_fence.add_trace(go.Surface(
            x=XF2, y=YF2, z=ZF2,
            surfacecolor=face,
            colorscale=COLORSCALE,
            cmin=cmin, cmax=cmax,
            showscale=not cb_f,
            colorbar=colorbar_cfg if not cb_f else None,
            name=f"{line_name} (Y={y_pos} m)",
            opacity=0.95,
        ))
        cb_f = True

        # Label
        fig_fence.add_trace(go.Scatter3d(
            x=[(xi[0]+xi[-1])/2],
            y=[float(y_pos)],
            z=[zi[-1] + abs(zi[-1]-zi[0])*0.18],
            mode='text',
            text=[f"<b>{line_name}<br>Y={y_pos} m</b>"],
            textfont=dict(size=11, color='black'),
            showlegend=False,
        ))

    fig_fence.update_layout(
        scene=dict(
            xaxis=dict(title='X (m)'),
            yaxis=dict(title='Y (m)'),
            zaxis=dict(title='Kedalaman (m)'),
            camera=dict(eye=dict(x=1.8, y=-1.6, z=1.0)),
            aspectmode='manual',
            aspectratio=dict(x=2.2, y=0.6, z=0.5),
        ),
        height=580,
        title=dict(text="FENCE DIAGRAM (PENAMPANG 3D)", x=0.5, font=dict(size=16)),
        margin=dict(l=0, r=20, t=50, b=0),
        paper_bgcolor='white',
    )
    st.plotly_chart(fig_fence, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — IRISAN HORIZONTAL
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Irisan Horizontal (pada Kedalaman Tertentu)")

    depth_opts = [f"{z:.2f} m" for z in zi]
    col1, col2, col3 = st.columns(3)
    with col1:
        d1_idx = st.selectbox("Kedalaman 1", range(len(zi)), format_func=lambda i: depth_opts[i], index=0)
    with col2:
        d2_idx = st.selectbox("Kedalaman 2", range(len(zi)), format_func=lambda i: depth_opts[i], index=len(zi)//2)
    with col3:
        d3_idx = st.selectbox("Kedalaman 3", range(len(zi)), format_func=lambda i: depth_opts[i], index=len(zi)-1)

    sel_idxs = [d1_idx, d2_idx, d3_idx]
    titles = [f"Kedalaman {zi[i]:.2f} m" for i in sel_idxs]

    fig_h = make_subplots(rows=1, cols=3, subplot_titles=titles,
                          horizontal_spacing=0.08)

    for col_i, iz_idx in enumerate(sel_idxs, 1):
        slice_h = plot_grid[:, :, iz_idx]   # NX x NY
        XH, YH  = np.meshgrid(xi, yi, indexing='ij')

        fig_h.add_trace(
            go.Contour(
                x=xi, y=yi, z=slice_h.T,
                colorscale=COLORSCALE,
                zmin=cmin, zmax=cmax,
                showscale=(col_i == 3),
                colorbar=dict(title=cbar_title, tickvals=tickvals, ticktext=ticktext) if col_i == 3 else None,
                contours=dict(coloring='heatmap', showlabels=True,
                              labelfont=dict(size=9, color='white')),
                hovertemplate='X:%{x:.1f} m<br>Y:%{y:.1f} m<br>'+cbar_title+':%{z:.3f}<extra></extra>',
            ), row=1, col=col_i
        )
        # Posisi lintasan
        for _, grow in geom.iterrows():
            fig_h.add_trace(go.Scatter(
                x=[xi[0], xi[-1]],
                y=[grow['Y_Position'], grow['Y_Position']],
                mode='lines+text',
                line=dict(color='white', width=1.5, dash='dot'),
                text=[grow['Line'], ''],
                textposition='top left',
                textfont=dict(size=9, color='white'),
                showlegend=False,
            ), row=1, col=col_i)

    fig_h.update_xaxes(title_text="X (m)")
    fig_h.update_yaxes(title_text="Y (m)")
    fig_h.update_layout(
        height=380,
        title=dict(text="IRISAN HORIZONTAL (KEDALAMAN TERTENTU)", x=0.5, font=dict(size=15)),
        paper_bgcolor='white',
    )
    st.plotly_chart(fig_h, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — IRISAN VERTIKAL
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Irisan Vertikal — Penampang Memanjang")

    col_a, col_b = st.columns([1, 3])
    with col_a:
        mode_v = st.radio("Tampilkan", ["Semua Lintasan", "Pilih Lintasan"])
        if mode_v == "Pilih Lintasan":
            sel_line = st.selectbox("Lintasan", geom['Line'].tolist())
        else:
            sel_line = None

    lines_show = geom['Line'].tolist() if mode_v == "Semua Lintasan" else [sel_line]
    ncols_v = len(lines_show)
    titles_v = [f"{l} (Y={line_y[l]} m)" for l in lines_show]

    fig_v = make_subplots(rows=1, cols=ncols_v, subplot_titles=titles_v,
                          horizontal_spacing=0.06)

    for col_i, lname in enumerate(lines_show, 1):
        iy = np.argmin(np.abs(yi - line_y[lname]))
        slice_v = plot_grid[:, iy, :]   # NX x NZ

        fig_v.add_trace(
            go.Contour(
                x=xi, y=zi, z=slice_v.T,
                colorscale=COLORSCALE,
                zmin=cmin, zmax=cmax,
                showscale=(col_i == ncols_v),
                colorbar=dict(title=cbar_title, tickvals=tickvals, ticktext=ticktext) if col_i == ncols_v else None,
                contours=dict(coloring='heatmap', showlabels=True,
                              labelfont=dict(size=9, color='white')),
                hovertemplate='X:%{x:.1f} m<br>Z:%{y:.2f} m<br>'+cbar_title+':%{z:.3f}<extra></extra>',
            ), row=1, col=col_i
        )

    fig_v.update_xaxes(title_text="X (m)")
    fig_v.update_yaxes(title_text="Kedalaman (m)")
    fig_v.update_layout(
        height=380,
        title=dict(text="IRISAN VERTIKAL (PENAMPANG MEMANJANG)", x=0.5, font=dict(size=15)),
        paper_bgcolor='white',
    )
    st.plotly_chart(fig_v, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — DATA & STATISTIK
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Ringkasan Data & Interpretasi Litologi")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Data Poin", len(df))
    c2.metric("Rentang Resistivitas", f"{df['Resistivity'].min():.2f} – {df['Resistivity'].max():.2f} Ω·m")
    c3.metric("Rentang Kedalaman", f"{df['Depth'].min():.2f} – {df['Depth'].max():.2f} m")

    st.markdown("---")
    st.markdown("### 🗺️ Legenda Interpretasi Litologi Gambut")

    interpretasi = [
        ("#00008B", "< 1 Ω·m",    "Lempung sangat jenuh / air"),
        ("#00AA44", "1 – 5 Ω·m",  "Gambut jenuh"),
        ("#CCCC00", "5 – 20 Ω·m", "Gambut kurang jenuh"),
        ("#006600", "20 – 100 Ω·m","Lempung"),
        ("#0000FF", "> 100 Ω·m",  "Pasir / kerikil"),
    ]
    cols_leg = st.columns(len(interpretasi))
    for ci, (color, rng, desc) in zip(cols_leg, interpretasi):
        ci.markdown(
            f'<div style="background:{color};padding:8px;border-radius:6px;text-align:center;">'
            f'<span style="color:white;font-weight:bold;font-size:12px;">{rng}<br>{desc}</span></div>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("### 📋 Statistik Per Lintasan")
    stats = df.groupby('Line')['Resistivity'].agg(['min','max','mean','std']).round(3)
    stats.columns = ['Min (Ω·m)','Max (Ω·m)','Rata-rata (Ω·m)','Std Dev']
    st.dataframe(stats, use_container_width=True)

    st.markdown("---")
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='DATA', index=False)
        geom.to_excel(writer, sheet_name='GEOMETRY', index=False)
        stats.to_excel(writer, sheet_name='STATISTIK')
    st.download_button(
        "⬇️ Download Data Lengkap (.xlsx)",
        data=buf.getvalue(),
        file_name="Geolistrik_3D_Hasil.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

st.markdown("---")
st.caption("📌 Visualisasi 3D Geolistrik | Interpolasi: Scipy griddata (linear + nearest fill) | Plotting: Plotly")
