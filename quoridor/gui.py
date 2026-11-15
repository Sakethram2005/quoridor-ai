"""Pygame front-end for the Quoridor agent.

Controls
--------
Mouse  : click a highlighted cell to move; hover near a groove between cells to
         preview a wall (green = legal, red = illegal) and click to place it.
Dropdowns (Blue / Red): pick Human or any AI (Minimax / MCTS level 1-4,
         Greedy, Random) for each side independently; changing one restarts the game.
N      : new game          S : swap sides
P      : show/hide A* shortest paths             ESC : quit

The AI thinks in a background thread so the window never freezes.
"""
from __future__ import annotations

import math
import random
import sys
import threading
import time
import traceback

import pygame

from .game import GOAL_ROW, N, State, describe_move
from .players import PLAYER_KEYS, PLAYER_LABELS, make_agent, preset_players

# ---------------------------------------------------------------- layout
CELL, GAP = 60, 14
PITCH = CELL + GAP
BOARD_PX = N * CELL + (N - 1) * GAP
OX = OY = 40
PANEL_X = OX + BOARD_PX + 36
WIN_W, WIN_H = PANEL_X + 290, BOARD_PX + 2 * OY
HIT_TOL = 5                       # extra pixels around a groove that count as a wall click
MIN_AI_DELAY = 0.45               # so AI-vs-AI moves are watchable

# ---------------------------------------------------------------- colours
BG = (22, 24, 31)
BOARD_BG = (34, 38, 50)
CELL_COL = (66, 74, 98)
GOAL_TINT = ((52, 82, 120), (112, 56, 62))      # goal row of player 0 / player 1
WALL_COL = (238, 192, 92)
PAWN_COL = ((92, 168, 255), (242, 96, 96))
TEXT = (232, 234, 242)
MUTED = (140, 146, 166)
GOOD = (96, 212, 132)
BAD = (236, 84, 84)
BTN, BTN_HOVER = (52, 58, 78), (72, 80, 108)

DD_ITEM_H = 26
PLAYER_NAMES = ("Blue (bottom)", "Red (top)")


def cell_rect(r, c):
    return pygame.Rect(OX + c * PITCH, OY + r * PITCH, CELL, CELL)


def cell_center(r, c):
    return cell_rect(r, c).center


def wall_rect(r, c, o):
    if o == "H":
        return pygame.Rect(OX + c * PITCH, OY + r * PITCH + CELL, 2 * CELL + GAP, GAP)
    return pygame.Rect(OX + c * PITCH + CELL, OY + r * PITCH, GAP, 2 * CELL + GAP)


class _Job(threading.Thread):
    """Runs agent.choose_move in the background."""

    def __init__(self, agent, state):
        super().__init__(daemon=True)
        self.agent, self.state = agent, state
        self.move, self.info, self.done = None, None, False
        self.earliest = time.time() + MIN_AI_DELAY

    def run(self):
        try:
            self.move = self.agent.choose_move(self.state)
            self.info = dict(getattr(self.agent, "last_stats", {}) or {})
            self.info["agent"] = self.agent.name
        except Exception:                                   # never kill the UI
            traceback.print_exc()
            cells = self.state.pawn_moves()
            self.move = ("move",) + random.choice(cells)
        self.done = True


class Button:
    def __init__(self, rect, label_fn, callback):
        self.rect, self.label_fn, self.callback = pygame.Rect(rect), label_fn, callback

    def draw(self, surf, font, mouse):
        col = BTN_HOVER if self.rect.collidepoint(mouse) else BTN
        pygame.draw.rect(surf, col, self.rect, border_radius=8)
        txt = font.render(self.label_fn(), True, TEXT)
        surf.blit(txt, txt.get_rect(center=self.rect.center))


class Dropdown:
    """Simple dropdown: click the header to open, click an item to choose."""

    def __init__(self, rect, keys, labels, value, on_change):
        self.rect = pygame.Rect(rect)
        self.keys, self.labels, self.value, self.on_change = keys, labels, value, on_change
        self.open = False

    def item_rect(self, i):
        return pygame.Rect(self.rect.x, self.rect.bottom + 2 + i * DD_ITEM_H, self.rect.w, DD_ITEM_H)

    def list_rect(self):
        return pygame.Rect(self.rect.x, self.rect.bottom + 2, self.rect.w, DD_ITEM_H * len(self.keys))

    def handle_click(self, pos):
        """Returns True if the click was consumed."""
        if self.open:
            for i, k in enumerate(self.keys):
                if self.item_rect(i).collidepoint(pos):
                    self.open = False
                    if k != self.value:
                        self.value = k
                        self.on_change()
                    return True
            self.open = False
            return self.rect.collidepoint(pos)       # clicking the header again just closes
        if self.rect.collidepoint(pos):
            self.open = True
            return True
        return False

    def draw_header(self, surf, font, mouse):
        col = BTN_HOVER if self.rect.collidepoint(mouse) or self.open else BTN
        pygame.draw.rect(surf, col, self.rect, border_radius=6)
        txt = font.render(self.labels[self.value], True, TEXT)
        surf.blit(txt, (self.rect.x + 10, self.rect.centery - txt.get_height() // 2))
        cx, cy = self.rect.right - 14, self.rect.centery
        tri = [(cx - 5, cy - 3), (cx + 5, cy - 3), (cx, cy + 4)]
        pygame.draw.polygon(surf, MUTED, tri)

    def draw_list(self, surf, font, mouse):
        if not self.open:
            return
        pygame.draw.rect(surf, (30, 33, 44), self.list_rect().inflate(4, 4), border_radius=6)
        for i, k in enumerate(self.keys):
            r = self.item_rect(i)
            if r.collidepoint(mouse):
                pygame.draw.rect(surf, BTN_HOVER, r, border_radius=4)
            colr = GOOD if k == self.value else TEXT
            txt = font.render(self.labels[k], True, colr)
            surf.blit(txt, (r.x + 10, r.centery - txt.get_height() // 2))


class App:
    def __init__(self, mode="human-ai", level=2, blue=None, red=None):
        pygame.init()
        pygame.display.set_caption("Quoridor - A* + Minimax")
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        self.clock = pygame.time.Clock()
        self.f_title = pygame.font.SysFont("segoeui,helvetica,arial", 34, bold=True)
        self.f_big = pygame.font.SysFont("segoeui,helvetica,arial", 22, bold=True)
        self.f = pygame.font.SysFont("segoeui,helvetica,arial", 18)
        self.f_small = pygame.font.SysFont("segoeui,helvetica,arial", 15)
        self.show_path = False
        d_blue, d_red = preset_players(mode, level)
        blue, red = blue or d_blue, red or d_red
        # one dropdown per side; y matches the player blocks drawn in draw_panel
        self.dropdowns = [
            Dropdown((PANEL_X + 28, OY + 78 + 24, 250, 26), PLAYER_KEYS, PLAYER_LABELS, blue, self.new_game),
            Dropdown((PANEL_X + 28, OY + 78 + 104 + 24, 250, 26), PLAYER_KEYS, PLAYER_LABELS, red, self.new_game),
        ]
        by = 560
        self.buttons = [
            Button((PANEL_X, by, 134, 38), lambda: "New game (N)", self.new_game),
            Button((PANEL_X + 144, by, 134, 38), lambda: "A* path (P)", self.toggle_path),
            Button((PANEL_X, by + 48, 278, 38), lambda: "Swap sides (S)", self.swap_sides),
        ]
        self.new_game()

    # ------------------------------------------------------------ game flow
    def new_game(self):
        self.state = State()
        self.history = []
        self.job = None
        self.ai_info = None
        self.controllers = self._make_controllers()

    def _make_controllers(self):
        # fresh agents every game (a stale search thread can't touch the new ones)
        return [make_agent(d.value) for d in self.dropdowns]

    def swap_sides(self):
        a, b = self.dropdowns
        a.value, b.value = b.value, a.value
        self.new_game()

    def toggle_path(self):
        self.show_path = not self.show_path

    def play(self, move, info=None):
        p = self.state.turn
        self.state = self.state.apply(move)
        self.history.append((p, move))
        if info:
            self.ai_info = info

    def update(self):
        if self.state.winner() is not None:
            return
        ctrl = self.controllers[self.state.turn]
        if ctrl is None:
            return
        if self.job is None:
            self.job = _Job(ctrl, self.state)
            self.job.start()
        elif self.job.done and time.time() >= self.job.earliest:
            move, info = self.job.move, self.job.info
            self.job = None
            if move is not None:
                self.play(move, info)

    def human_turn(self):
        return self.state.winner() is None and self.controllers[self.state.turn] is None

    # ------------------------------------------------------------- picking
    def pick_target(self, pos):
        mx, my = pos[0] - OX, pos[1] - OY
        if not (-6 <= mx <= BOARD_PX + 6 and -6 <= my <= BOARD_PX + 6):
            return None
        best = None
        g = round((my - CELL - GAP / 2) / PITCH)                  # horizontal groove index
        if 0 <= g <= N - 2:
            d = abs(my - (g * PITCH + CELL + GAP / 2))
            if d <= GAP / 2 + HIT_TOL:
                c = min(N - 2, max(0, math.floor(mx / PITCH - 0.5)))
                best = (d, ("wall", g, c, "H"))
        h = round((mx - CELL - GAP / 2) / PITCH)                  # vertical groove index
        if 0 <= h <= N - 2:
            d = abs(mx - (h * PITCH + CELL + GAP / 2))
            if d <= GAP / 2 + HIT_TOL:
                r = min(N - 2, max(0, math.floor(my / PITCH - 0.5)))
                if best is None or d < best[0]:
                    best = (d, ("wall", r, h, "V"))
        if best:
            return best[1]
        col, row = int(mx // PITCH), int(my // PITCH)
        if 0 <= row < N and 0 <= col < N and mx - col * PITCH <= CELL and my - row * PITCH <= CELL:
            return ("cell", row, col)
        return None

    def click(self, pos):
        was_open = any(d.open for d in self.dropdowns)
        for d in self.dropdowns:                          # an open list sits on top
            if d.open and d.handle_click(pos):
                return
        if was_open:                                      # that click only closed the list
            return
        for d in self.dropdowns:
            if d.handle_click(pos):
                return
        for b in self.buttons:
            if b.rect.collidepoint(pos):
                b.callback()
                return
        if not self.human_turn():
            return
        t = self.pick_target(pos)
        if t is None:
            return
        if t[0] == "cell":
            if (t[1], t[2]) in self.state.pawn_moves():
                self.play(("move", t[1], t[2]))
        elif self.state.try_wall(t[1], t[2], t[3]) is not None:
            self.play(t)

    # ------------------------------------------------------------- drawing
    def draw_board(self, mouse):
        s = self.state
        pygame.draw.rect(self.screen, BOARD_BG,
                         (OX - 14, OY - 14, BOARD_PX + 28, BOARD_PX + 28), border_radius=14)
        for r in range(N):
            for c in range(N):
                col = CELL_COL
                if r == GOAL_ROW[0]:
                    col = GOAL_TINT[0]
                elif r == GOAL_ROW[1]:
                    col = GOAL_TINT[1]
                pygame.draw.rect(self.screen, col, cell_rect(r, c), border_radius=8)
        for i in range(N):                                         # coordinates
            t = self.f_small.render("abcdefghi"[i], True, MUTED)
            self.screen.blit(t, t.get_rect(center=(cell_rect(0, i).centerx, OY + BOARD_PX + 20)))
            t = self.f_small.render(str(N - i), True, MUTED)
            self.screen.blit(t, t.get_rect(center=(OX - 24, cell_rect(i, 0).centery)))

        if self.show_path:                                         # A* overlay
            for p in (0, 1):
                path = s.shortest_path(p)
                if path:
                    pts = [cell_center(*cell) for cell in path]
                    if len(pts) > 1:
                        pygame.draw.lines(self.screen, PAWN_COL[p], False, pts, 4)
                    for pt in pts[1:]:
                        pygame.draw.circle(self.screen, PAWN_COL[p], pt, 5)

        if self.human_turn():                                      # legal destinations
            for r, c in s.pawn_moves():
                pygame.draw.circle(self.screen, GOOD, cell_center(r, c), 9)

        for r, c, o in s.walls:                                    # placed walls
            pygame.draw.rect(self.screen, WALL_COL, wall_rect(r, c, o), border_radius=5)

        for p in (0, 1):                                           # pawns
            cx, cy = cell_center(*s.pawns[p])
            if s.turn == p and s.winner() is None:
                pygame.draw.circle(self.screen, (255, 255, 255), (cx, cy), 28, 3)
            pygame.draw.circle(self.screen, PAWN_COL[p], (cx, cy), 23)
            pygame.draw.circle(self.screen, (20, 20, 28), (cx, cy), 23, 2)

        if self.human_turn():                                      # wall preview
            t = self.pick_target(mouse)
            if t and t[0] == "wall":
                ok = s.try_wall(t[1], t[2], t[3]) is not None
                rect = wall_rect(*t[1:])
                ghost = pygame.Surface(rect.size, pygame.SRCALPHA)
                ghost.fill((*(GOOD if ok else BAD), 170))
                self.screen.blit(ghost, rect)

    def draw_panel(self, mouse):
        s = self.state
        x = PANEL_X
        self.screen.blit(self.f_title.render("QUORIDOR", True, TEXT), (x, OY - 8))
        self.screen.blit(self.f_small.render("A* path-finding + Minimax (alpha-beta)", True, MUTED),
                         (x, OY + 34))
        y = OY + 78
        for p in (0, 1):
            pygame.draw.circle(self.screen, PAWN_COL[p], (x + 10, y + 12), 10)
            self.screen.blit(self.f_big.render(PLAYER_NAMES[p], True, TEXT), (x + 28, y))
            self.dropdowns[p].draw_header(self.screen, self.f_small, mouse)      # who plays this side
            d = s.dist(p)
            self.screen.blit(self.f_small.render(f"A* distance to goal: {d}", True, TEXT), (x + 28, y + 54))
            for i in range(10):                                    # walls left
                col = WALL_COL if i < s.walls_left[p] else (60, 64, 80)
                pygame.draw.rect(self.screen, col, (x + 28 + i * 24, y + 78, 18, 7), border_radius=3)
            y += 104

        w = s.winner()
        if w is not None:
            msg, col = f"{PLAYER_NAMES[w]} wins!  (N = new game)", GOOD
        elif self.controllers[s.turn] is None:
            msg, col = f"Turn: {PLAYER_NAMES[s.turn]} - your move", TEXT
        else:
            msg, col = f"Turn: {PLAYER_NAMES[s.turn]} - AI thinking...", WALL_COL
        self.screen.blit(self.f.render(msg, True, col), (x, y))
        y += 32

        if self.history:
            p, mv = self.history[-1]
            self.screen.blit(self.f_small.render(f"Last: {PLAYER_NAMES[p].split()[0]} {describe_move(mv)}",
                                                 True, MUTED), (x, y))
        y += 22
        if self.ai_info and "depth" in self.ai_info:
            i = self.ai_info
            self.screen.blit(self.f_small.render(
                f"{i['agent']}: depth {i['depth']}, {i['nodes']} nodes, {i['time']:.2f}s", True, MUTED), (x, y))
        y += 22
        self.screen.blit(self.f_small.render(f"Moves played: {len(self.history)}", True, MUTED), (x, y))

        for b in self.buttons:
            b.draw(self.screen, self.f, mouse)
        for d in self.dropdowns:                                   # open lists are drawn last (on top)
            d.draw_list(self.screen, self.f_small, mouse)

    # ------------------------------------------------------------ main loop
    def run(self):
        running = True
        while running:
            mouse = pygame.mouse.get_pos()
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_ESCAPE:
                        running = False
                    elif e.key == pygame.K_n:
                        self.new_game()
                    elif e.key == pygame.K_s:
                        self.swap_sides()
                    elif e.key == pygame.K_p:
                        self.toggle_path()
                elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                    self.click(e.pos)
            self.update()
            self.screen.fill(BG)
            self.draw_board(mouse)
            self.draw_panel(mouse)
            pygame.display.flip()
            self.clock.tick(30)
        pygame.quit()


def run(mode="human-ai", level=2, blue=None, red=None):
    App(mode, level, blue, red).run()