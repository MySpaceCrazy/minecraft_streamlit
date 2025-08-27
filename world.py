import numpy as np

class World:
    def __init__(self, width=20, height=10):
        # Matriz 2D representando o mundo
        # 0 = vazio, 1 = bloco
        self.grid = np.zeros((height, width), dtype=int)

        # Posição inicial do jogador
        self.player_x = width // 2
        self.player_y = height - 1  # começa no "chão"

        # Velocidade de movimento
        self.is_running = False

    def toggle_run(self):
        """Alterna entre andar e correr"""
        self.is_running = not self.is_running

    def move(self, direction):
        """Move o jogador para esquerda/direita"""
        step = 2 if self.is_running else 1
        if direction == "left":
            self.player_x = max(0, self.player_x - step)
        elif direction == "right":
            self.player_x = min(self.grid.shape[1] - 1, self.player_x + step)

    def jump(self):
        """Pular: só se estiver em chão/bloco"""
        if self.player_y == self.grid.shape[0] - 1 or self.grid[self.player_y + 1, self.player_x] == 1:
            if self.player_y > 0:
                self.player_y -= 1  # sobe 1 bloco

    def gravity(self):
        """Aplica gravidade (cai se não tiver chão)"""
        if self.player_y < self.grid.shape[0] - 1:
            if self.grid[self.player_y + 1, self.player_x] == 0:
                self.player_y += 1

    def place_block(self):
        """Coloca bloco abaixo do jogador"""
        if self.player_y + 1 < self.grid.shape[0]:
            self.grid[self.player_y + 1, self.player_x] = 1

    def break_block(self):
        """Destrói bloco abaixo do jogador"""
        if self.player_y + 1 < self.grid.shape[0]:
            self.grid[self.player_y + 1, self.player_x] = 0

    def get_state(self):
        """Retorna o estado do mundo (inclui jogador)"""
        world_copy = self.grid.copy()
        world_copy[self.player_y, self.player_x] = 2  # 2 = jogador
        return world_copy
