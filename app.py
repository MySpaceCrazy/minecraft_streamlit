import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple

try:
    # Opcional, melhora teclado: mapeia atalhos para os botões
    # pip install streamlit-shortcuts
    from streamlit_shortcuts import add_shortcuts
    SHORTCUTS_OK = True
except Exception:
    SHORTCUTS_OK = False

# =============================
# Config & Constantes do "motor"
# =============================
WORLD_W, WORLD_H = 64, 36       # largura x altura do mundo (em tiles)
TICKS_PER_SEC = 30              # taxa de atualização (via autorefresh)
GRAVITY = 28.0                  # gravidade (tiles/seg^2)
JUMP_VEL = 12.5                 # velocidade do pulo
WALK_SPEED = 6.0                # andar (tiles/seg)
RUN_SPEED = 11.0                # correr (tiles/seg)
PLAYER_W, PLAYER_H = 0.9, 1.8   # tamanho do jogador (em tiles)

# Tipos de bloco (valores do grid)
AIR   = 0
GRASS = 1
DIRT  = 2
BRICK = 3
SAND  = 4
STONE = 5

PALETTE = {
    AIR:   (1.0, 1.0, 1.0),  # branco (fundo)
    GRASS: (0.35, 0.75, 0.35),
    DIRT:  (0.55, 0.35, 0.2),
    BRICK: (0.7, 0.2, 0.2),
    SAND:  (0.9, 0.85, 0.55),
    STONE: (0.55, 0.55, 0.6),
}
PLAYER_COLOR = (0.1, 0.1, 0.9)
CURSOR_COLOR = (0.0, 0.0, 0.0)

BLOCK_NAMES = {
    GRASS: "Grama",
    DIRT:  "Terra",
    BRICK: "Tijolo",
    SAND:  "Areia",
    STONE: "Pedra",
}
INVENTORY = [BRICK, GRASS, SAND, STONE, DIRT]

@dataclass
class Player:
    x: float = 8.0
    y: float = 18.0
    vx: float = 0.0
    vy: float = 0.0
    facing: int = 1   # 1 direita, -1 esquerda
    running: bool = False
    on_ground: bool = False
    sel_idx: int = 0  # índice do bloco selecionado no inventário

# =============================
# Mundo
# =============================

def gen_world(w: int, h: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    world = np.zeros((h, w), dtype=np.int8)

    # terreno básico (tipo terraria): chão irregular + camadas
    base = int(h * 0.25)
    heightmap = base + (rng.integers(-3, 4, size=w))
    heightmap = np.clip(heightmap, 3, h - 6)
    for x in range(w):
        y0 = heightmap[x]
        world[:y0-1, x] = AIR
        world[y0-1, x] = GRASS
        world[y0:y0+3, x] = DIRT
        world[y0+3:, x] = STONE
    
    # alguns veios de areia
    for _ in range(10):
        cx, cy = rng.integers(4, w-4), rng.integers(base+2, h-6)
        radx, rady = rng.integers(2, 5), rng.integers(1, 3)
        for x in range(cx - radx, cx + radx + 1):
            for y in range(cy - rady, cy + rady + 1):
                if 0 <= x < w and 0 <= y < h and ((x - cx)**2)/(radx**2) + ((y - cy)**2)/(rady**2) <= 1.0:
                    if world[y, x] != AIR:
                        world[y, x] = SAND

    # muralhas laterais
    world[:, 0] = STONE
    world[:, -1] = STONE
    world[0, :] = STONE
    world[-1, :] = STONE

    return world

# =============================
# Física & colisões
# =============================

def rect_collides(world: np.ndarray, x: float, y: float, w: float, h: float) -> bool:
    """Verifica colisão AABB do retângulo com blocos sólidos."""
    x0, x1 = int(np.floor(x - w/2)), int(np.floor(x + w/2))
    y0, y1 = int(np.floor(y)), int(np.floor(y + h - 0.001))
    H, W = world.shape
    x0, x1 = max(0, x0), min(W-1, x1)
    y0, y1 = max(0, y0), min(H-1, y1)
    for yy in range(y0, y1+1):
        for xx in range(x0, x1+1):
            if world[yy, xx] != AIR:
                return True
    return False


def step_physics(world: np.ndarray, p: Player, dt: float, move_left: bool, move_right: bool, jump: bool):
    # velocidade horizontal
    target_v = 0.0
    if move_left:
        target_v -= RUN_SPEED if p.running else WALK_SPEED
        p.facing = -1
    if move_right:
        target_v += RUN_SPEED if p.running else WALK_SPEED
        p.facing = 1

    p.vx = target_v

    # pulo
    if jump and p.on_ground:
        p.vy = JUMP_VEL
        p.on_ground = False

    # gravidade
    p.vy -= GRAVITY * dt

    # Integrar e resolver colisões separadamente (X depois Y)
    # --- eixo X
    new_x = p.x + p.vx * dt
    if rect_collides(world, new_x, p.y, PLAYER_W, PLAYER_H):
        # encosta na parede -> para horizontal
        # empurra até borda do bloco
        step = np.sign(p.vx) * 0.01
        while not rect_collides(world, p.x + step, p.y, PLAYER_W, PLAYER_H):
            p.x += step
        p.vx = 0.0
    else:
        p.x = new_x

    # --- eixo Y
    new_y = p.y + p.vy * dt
    if rect_collides(world, p.x, new_y, PLAYER_W, PLAYER_H):
        # chão ou teto
        if p.vy < 0:
            p.on_ground = True
        p.vy = 0.0
        step = np.sign(new_y - p.y) * 0.01
        while not rect_collides(world, p.x, p.y + step, PLAYER_W, PLAYER_H):
            p.y += step
    else:
        p.y = new_y
        p.on_ground = False

    # limitar dentro do mundo
    p.x = float(np.clip(p.x, 1.5, WORLD_W - 1.5))
    p.y = float(np.clip(p.y, 1.0, WORLD_H - 2.5))

# =============================
# Construção / destruição
# =============================

def target_tile(p: Player) -> Tuple[int, int]:
    tx = int(round(p.x + p.facing * 1.0))
    ty = int(round(p.y + 0.2))
    return tx, ty


def place_block(world: np.ndarray, p: Player):
    tx, ty = target_tile(p)
    if 0 <= tx < world.shape[1] and 0 <= ty < world.shape[0]:
        if world[ty, tx] == AIR:
            b = INVENTORY[p.sel_idx]
            # não deixar bloquear dentro do jogador
            px0, px1 = p.x - PLAYER_W/2, p.x + PLAYER_W/2
            py0, py1 = p.y, p.y + PLAYER_H
            if not (px0 <= tx <= px1 and py0 <= ty <= py1):
                world[ty, tx] = b


def destroy_block(world: np.ndarray, p: Player):
    tx, ty = target_tile(p)
    if 0 <= tx < world.shape[1] and 0 <= ty < world.shape[0]:
        if world[ty, tx] != AIR:
            world[ty, tx] = AIR

# =============================
# Render
# =============================

def render_world(world: np.ndarray, p: Player):
    H, W = world.shape
    rgb = np.zeros((H, W, 3), dtype=float)
    for k, col in PALETTE.items():
        rgb[world == k] = col

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(np.flipud(rgb), interpolation='nearest', origin='lower')  # y pra cima

    # jogador
    px = p.x
    py = p.y
    rect_x = [px - PLAYER_W/2, px + PLAYER_W/2, px + PLAYER_W/2, px - PLAYER_W/2, px - PLAYER_W/2]
    rect_y = [py, py, py + PLAYER_H, py + PLAYER_H, py]
    ax.plot(rect_x, rect_y)
    ax.fill_between(rect_x[:2], rect_y[:2], rect_y[2:4], step='pre', alpha=0.9)

    # cursor (tile alvo)
    tx, ty = target_tile(p)
    if 0 <= tx < W and 0 <= ty < H:
        cx = [tx - 0.5, tx + 0.5, tx + 0.5, tx - 0.5, tx - 0.5]
        cy = [ty - 0.5, ty - 0.5, ty + 0.5, ty + 0.5, ty - 0.5]
        ax.plot(cx, cy)

    ax.set_title("W/A/D para mover e pular | Shift para correr | F construir | G destruir | 1-5 troca bloco | R reset")
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_xticks([])
    ax.set_yticks([])
    st.pyplot(fig, clear_figure=True)

# =============================
# UI / Estado
# =============================

st.set_page_config(page_title="Voxel 2D — Streamlit", layout="wide")

if "world" not in st.session_state:
    st.session_state.world = gen_world(WORLD_W, WORLD_H, seed=42)
if "player" not in st.session_state:
    st.session_state.player = Player()
if "move_left" not in st.session_state:
    st.session_state.move_left = False
if "move_right" not in st.session_state:
    st.session_state.move_right = False
if "jump_req" not in st.session_state:
    st.session_state.jump_req = False

# Painel lateral (HUD)
st.sidebar.header("Inventário & Controles")
colA, colB = st.sidebar.columns(2)
with colA:
    if st.button("◀ Esquerda", use_container_width=True):
        st.session_state.move_left = not st.session_state.move_left
with colB:
    if st.button("Direita ▶", use_container_width=True):
        st.session_state.move_right = not st.session_state.move_right

col1, col2, col3 = st.sidebar.columns(3)
with col1:
    if st.button("Pular", use_container_width=True):
        st.session_state.jump_req = True
with col2:
    st.session_state.player.running = st.toggle("Correr", value=st.session_state.player.running)
with col3:
    if st.button("Reset", use_container_width=True):
        st.session_state.world = gen_world(WORLD_W, WORLD_H, seed=np.random.randint(0, 10_000))
        st.session_state.player = Player()

st.sidebar.subheader("Bloco selecionado")
sel = st.segmented_control(" ", options=list(range(len(INVENTORY))), format_func=lambda i: BLOCK_NAMES[INVENTORY[i]], key="inv_sel")
st.session_state.player.sel_idx = int(sel)

# Teclado (atalhos para os botões/toggles)
if SHORTCUTS_OK:
    add_shortcuts(
        # movimento
        **{
            # setas ou A/D
            "◀ Esquerda": ["a", "arrowleft"],
            "Direita ▶": ["d", "arrowright"],
            "Pular": ["w", "space"],
            # correr como toggle no painel
        }
    )

# Ações de construir / destruir por botões + atalhos
b1, b2, b3 = st.columns([1,1,2])
with b1:
    build_clicked = st.button("F: Construir", use_container_width=True)
with b2:
    destroy_clicked = st.button("G: Destruir", use_container_width=True)
with b3:
    st.write(":grey[Use **1..5** na segmentação acima para trocar o bloco]")

if SHORTCUTS_OK:
    add_shortcuts(**{
        "F: Construir": "f",
        "G: Destruir": "g",
    })

# Processar ações instantâneas
if build_clicked:
    place_block(st.session_state.world, st.session_state.player)
if destroy_clicked:
    destroy_block(st.session_state.world, st.session_state.player)
if st.session_state.jump_req:
    # será consumido no step de física
    pass

# Tick (autorefresh)
st_autorefresh = st.sidebar.slider("FPS", 10, 60, value=TICKS_PER_SEC, help="Taxa de atualização da simulação")
st.runtime.legacy_caching.clear_cache() if False else None  # no-op apenas para silenciar linters
st.experimental_set_query_params(tick=np.random.randint(0, 1_000_000))  # força URL a mudar levemente (evita cache do navegador)
st_autorefresh_ms = int(1000 / st_autorefresh)
st_autorefresh_result = st.experimental_rerun if False else None  # apenas placeholder, não usado diretamente
st._legacy_add_rows = None  # nada

# Simular um frame
DT = 1.0 / st_autorefresh
p = st.session_state.player
step_physics(
    st.session_state.world,
    p,
    DT,
    move_left=st.session_state.move_left,
    move_right=st.session_state.move_right,
    jump=st.session_state.jump_req,
)
st.session_state.jump_req = False

# Renderizar
render_world(st.session_state.world, st.session_state.player)

# HUD final
info1, info2, info3 = st.columns(3)
with info1:
    st.metric("Posição", f"x={p.x:.2f}, y={p.y:.2f}")
with info2:
    st.metric("Velocidade", f"vx={p.vx:.2f}, vy={p.vy:.2f}")
with info3:
    st.metric("No chão?", "Sim" if p.on_ground else "Não")

st.caption(
    "W/A/D movem e pulam; Shift alterna 'Correr' (toggle no painel); F constrói; G destrói; 1–5 trocam bloco.\n"
    "Se instalou o pacote opcional **streamlit-shortcuts**, os atalhos de teclado funcionam de forma mais fluida."
)