import pygame
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import os
from collections import deque

# === CONFIGURAÇÕES ===
WIDTH = 800
HEIGHT = 600
BIRD_SIZE = 30
PIPE_WIDTH = 50
GAP_HEIGHT = 150
FPS = 60
NUM_BIRDS = 10

ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.1
EPISODES = 1000
VISUAL_MODE = False  # <- Toggle para ver ou não o jogo

JUMP_STRENGTH = -10
GRAVITY = 0.5

# === DQN CONFIG ===
BATCH_SIZE = 64
MEMORY_SIZE = 10000
LEARNING_RATE = 0.001
TARGET_UPDATE_FREQ = 10

# === MODELO DE REDE ===
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 128),  # 4 entradas: y, velocidade, distância do cano, gap
            nn.ReLU(),
            nn.Linear(128, 2)  # 2 ações: pular ou não
        )

    def forward(self, x):
        return self.model(x)

# === FUNÇÕES AUXILIARES ===
def get_state_continuous(bird_y, bird_vel, pipe_x, gap_y):
    # Normaliza as entradas (valores contínuos)
    norm_y = bird_y / HEIGHT
    norm_vel = bird_vel / 20
    norm_dist = (pipe_x - 50) / WIDTH
    norm_gap = (gap_y - bird_y) / HEIGHT
    return torch.tensor([norm_y, norm_vel, norm_dist, norm_gap], dtype=torch.float32)

def select_action(state, epsilon, policy_net):
    if random.random() < epsilon:
        return random.randint(0, 1)  # Ação aleatória
    with torch.no_grad():
        return torch.argmax(policy_net(state)).item()  # Ação com maior valor Q

def optimize_model(memory, policy_net, target_net, optimizer):
    if len(memory) < BATCH_SIZE:
        return
    batch = random.sample(memory, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.stack(states)
    actions = torch.tensor(actions).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
    next_states = torch.stack(next_states)
    dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

    q_values = policy_net(states).gather(1, actions)
    next_q_values = target_net(next_states).max(1)[0].detach().unsqueeze(1)
    expected_q_values = rewards + (GAMMA * next_q_values * (1 - dones))

    loss = nn.MSELoss()(q_values, expected_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# === INICIALIZAÇÃO ===
if VISUAL_MODE:
    pygame.init()
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('Arial', 24)
else:
    clock = pygame.time.Clock()

# === RECOMPENSA ===
def get_reward(alive, passed, just_died=False):
    if just_died:
        return -1000
    if passed:
        return 50
    return 1

# === DRAW ===
def draw_all(birds, pipe_x, gap_y, score):
    if not VISUAL_MODE:
        return
    win.fill((0, 0, 0))
    for i, (y, color) in enumerate(birds):
        pygame.draw.rect(win, color, (50, y, BIRD_SIZE, BIRD_SIZE))
    pygame.draw.rect(win, (0, 255, 0), (pipe_x, 0, PIPE_WIDTH, gap_y))
    pygame.draw.rect(win, (0, 255, 0), (pipe_x, gap_y + GAP_HEIGHT, PIPE_WIDTH, HEIGHT - gap_y - GAP_HEIGHT))
    text = font.render(f"Score: {score}", True, (255, 255, 255))
    win.blit(text, (10, 10))
    pygame.display.flip()

# === Q-TABLE SETUP ===
policy_net = DQN()
target_net = DQN()
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
memory = deque(maxlen=MEMORY_SIZE)

# === VARIÁVEIS PARA TREINAMENTO ===
save_path = "qtable_best_avg.pkl"
best_q_tables = []

if os.path.exists(save_path):
    with open(save_path, "rb") as f:
        best_q_tables = pickle.load(f)
    print("✅ Q-tables carregadas com sucesso!")

# === TREINAMENTO ===
try:
    for ep in range(EPISODES):
        print(f"\n=== Episódio {ep + 1} ===")
        max_score = 0
        best_q_for_episode = []

        bird_y = [HEIGHT // 2 for _ in range(NUM_BIRDS)]
        bird_vel = [0 for _ in range(NUM_BIRDS)]
        pipe_x = WIDTH
        gap_y = random.randint(100, HEIGHT - 200)
        score = 0
        alive = [True] * NUM_BIRDS
        colors = [tuple(np.random.randint(50, 255, size=3)) for _ in range(NUM_BIRDS)]

        while any(alive):
            clock.tick(60000 if not VISUAL_MODE else FPS)

            if VISUAL_MODE:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        with open("checkpoint.pkl", "wb") as f:
                            pickle.dump((policy_net.state_dict(), target_net.state_dict(), best_q_tables), f)
                        pygame.quit()
                        exit()

            pipe_x -= 10

            passou = False
            if pipe_x + PIPE_WIDTH < 0:
                pipe_x = WIDTH
                gap_y = random.randint(100, HEIGHT - 200)
                score += 1
                passou = True
                print(f"Pássaros vivos passaram por um cano! Score: {score}")
                if score > max_score:
                    max_score = score

            for i in range(NUM_BIRDS):
                if not alive[i]: continue

                state = get_state_continuous(bird_y[i], bird_vel[i], pipe_x, gap_y)
                action = select_action(state, EPSILON, policy_net)

                if action == 1:
                    bird_vel[i] = JUMP_STRENGTH
                bird_vel[i] += GRAVITY
                bird_y[i] += bird_vel[i]

                just_died = False

                if bird_y[i] < 0 or bird_y[i] + BIRD_SIZE > HEIGHT:
                    just_died = True
                    alive[i] = False
                elif pipe_x < 70 < pipe_x + PIPE_WIDTH:
                    if bird_y[i] < gap_y or bird_y[i] + BIRD_SIZE > gap_y + GAP_HEIGHT:
                        just_died = True
                        alive[i] = False

                reward = get_reward(alive[i], passou, just_died)
                next_state = get_state_continuous(bird_y[i], bird_vel[i], pipe_x, gap_y)

                memory.append((state, action, reward, next_state, float(alive[i])))

            optimize_model(memory, policy_net, target_net, optimizer)

            if ep % TARGET_UPDATE_FREQ == 0:
                target_net.load_state_dict(policy_net.state_dict())

            draw_all([(bird_y[i], colors[i]) for i in range(NUM_BIRDS) if alive[i]], pipe_x, gap_y, score)

        if not any(alive):
            print("⚠️ Nenhum pássaro sobreviveu...")
        
        print(f"🏁 Score do episódio: {max_score}")
        print(f"🎯 Média das melhores Q-tables: {np.mean([np.max(q) for q in best_q_tables]):.2f}")

        with open("checkpoint.pkl", "wb") as f:
            pickle.dump((policy_net.state_dict(), target_net.state_dict(), best_q_tables), f)

except KeyboardInterrupt:
    print("⛔ Interrompido manualmente. Salvando progresso...")
    with open(save_path, "wb") as f:
        pickle.dump(best_q_tables, f)
    pygame.quit()
    exit()

with open(save_path, "wb") as f:
    pickle.dump(best_q_tables, f)

print("✅ Q-tables salvas com sucesso!")

if VISUAL_MODE:
    pygame.quit()
