"""Pygame front-end for the Quoridor agent.

Controls
--------
Mouse  : click a highlighted cell to move; hover near a groove between cells to
         preview a wall (green = legal, red = illegal) and click to place it.
N      : new game          M : cycle mode        L : cycle AI level
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

from .agent import GreedyAgent, LEVELS, MinimaxAgent
from .game import GOAL_ROW, N, State, cell_name, describe_move
from .mcts import MCTSAgent

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

MODES = ["Human vs AI", "AI vs Greedy (demo)", "AI vs AI", "Human vs Human",
         "Human vs MCTS", "Minimax vs MCTS"]
MCTS_ITERS = {1: 150, 2: 400, 3: 1000, 4: 2500}      # AI level -> MCTS iterations per move
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


class App:
    def __init__(self, mode=0, level=2):
        pygame.init()
        pygame.display.set_caption("Quoridor - A* + Minimax")
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        self.clock = pygame.time.Clock()
        self.f_title = pygame.font.SysFont("segoeui,helvetica,arial", 34, bold=True)
        self.f_big = pygame.font.SysFont("segoeui,helvetica,arial", 22, bold=True)
        self.f = pygame.font.SysFont("segoeui,helvetica,arial", 18)
        self.f_small = pygame.font.SysFont("segoeui,helvetica,arial", 15)
        self.mode_idx, self.level = mode, level
        self.show_path = False
        by = 560
        self.buttons = [
            Button((PANEL_X, by, 134, 38), lambda: "New game (N)", self.new_game),
            Button((PANEL_X + 144, by, 134, 38), lambda: "A* path (P)", self.toggle_path),
            Button((PANEL_X, by + 48, 278, 38), lambda: f"Mode (M): {MODES[self.mode_idx]}", self.next_mode),
            Button((PANEL_X, by + 96, 278, 38), lambda: f"AI level (L): {self.level}  "
                   f"[d{LEVELS[self.level][0]} | {MCTS_ITERS[self.level]} it]", self.next_level),
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
        depth, tl = LEVELS[self.level]
        mm = lambda: MinimaxAgent(depth, tl)
        mc = lambda: MCTSAgent(MCTS_ITERS[self.level], time_limit=tl)
        mode = MODES[self.mode_idx]
        if mode == "Human vs MCTS":
            return [None, mc()]
        if mode == "Minimax vs MCTS":
            return [mm(), mc()]
        if mode == "Human vs AI":
            return [None, mm()]
        if mode == "AI vs Greedy (demo)":
            return [mm(), GreedyAgent()]
        if mode == "AI vs AI":
            return [mm(), mm()]
        return [None, None]

    def next_mode(self):
        self.mode_idx = (self.mode_idx + 1) % len(MODES)
        self.new_game()

    def next_level(self):
        self.level = self.level % max(LEVELS) + 1
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
            ctrl = self.controllers[p]
            who = "Human" if ctrl is None else ctrl.name
            pygame.draw.circle(self.screen, PAWN_COL[p], (x + 10, y + 12), 10)
            self.screen.blit(self.f_big.render(PLAYER_NAMES[p], True, TEXT), (x + 28, y))
            self.screen.blit(self.f_small.render(who, True, MUTED), (x + 28, y + 26))
            d = s.dist(p)
            self.screen.blit(self.f_small.render(f"A* distance to goal: {d}", True, TEXT), (x + 28, y + 46))
            for i in range(10):                                    # walls left
                col = WALL_COL if i < s.walls_left[p] else (60, 64, 80)
                pygame.draw.rect(self.screen, col, (x + 28 + i * 24, y + 70, 18, 7), border_radius=3)
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
                    elif e.key == pygame.K_m:
                        self.next_mode()
                    elif e.key == pygame.K_l:
                        self.next_level()
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


def run(mode=0, level=2):
    App(mode, level).run()
