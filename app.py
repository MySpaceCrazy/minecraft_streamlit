import streamlit as st
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="Minecraft Lite", layout="wide")

st.title("🟩 Minecraft Lite no Streamlit")

# Dimensão do mundo
size_x, size_y, size_z = 10, 10, 5

# Criação de blocos (1 = bloco, 0 = vazio)
world = np.zeros((size_x, size_y, size_z))

# Gerar um "terreno básico"
for x in range(size_x):
    for y in range(size_y):
        height = np.random.randint(1, 4)  # altura aleatória
        world[x, y, :height] = 1

# Criar lista de cubos (coordenadas)
x_coords, y_coords, z_coords = np.where(world == 1)

# Criar cubos no Plotly
fig = go.Figure()

for x, y, z in zip(x_coords, y_coords, z_coords):
    fig.add_trace(go.Mesh3d(
        x=[x, x+1, x+1, x, x, x+1, x+1, x],
        y=[y, y, y+1, y+1, y, y, y+1, y+1],
        z=[z, z, z, z, z+1, z+1, z+1, z+1],
        color='green',
        opacity=0.8,
        showscale=False
    ))

# Configuração da cena
fig.update_layout(
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
    ),
    margin=dict(r=0, l=0, b=0, t=0),
)

st.plotly_chart(fig, use_container_width=True)
