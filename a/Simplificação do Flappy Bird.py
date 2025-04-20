import numpy as np
import random

# Parâmetros do jogo
ALTURAS = 10
DISTANCIAS = 10
ACTIONS = 2  # 0 = não pular, 1 = pular

q_table = np.zeros((ALTURAS, DISTANCIAS, ACTIONS))

alpha = 0.1
gamma = 0.95
epsilon = 0.2
episodes = 500

def get_reward(alive, passed_pipe):
    if not alive:
        return -100
    return 1 + (10 if passed_pipe else 0)

# Simula o jogo
for ep in range(episodes):
    bird_y = 5
    pipe_dist = 9
    pipe_gap_y = random.randint(3, 6)
    done = False

    while not done:
        state = (bird_y, pipe_dist)

        # Escolhe ação
        if random.random() < epsilon:
            action = random.randint(0, 1)
        else:
            action = np.argmax(q_table[bird_y][pipe_dist])

        # Atualiza estado
        if action == 1:
            bird_y = max(0, bird_y - 2)  # pulo sobe
        else:
            bird_y = min(ALTURAS - 1, bird_y + 1)  # gravidade desce

        pipe_dist -= 1
        if pipe_dist < 0:
            pipe_dist = 9
            pipe_gap_y = random.randint(3, 6)

        # Verifica colisão
        alive = pipe_gap_y - 1 <= bird_y <= pipe_gap_y + 1
        alive = alive if pipe_dist != 0 else False  # só colide se estiver no cano

        reward = get_reward(alive, pipe_dist == 0 and alive)

        # Próximo estado
        next_state = (bird_y, pipe_dist)
        old_q = q_table[state][action]
        future_q = np.max(q_table[next_state])
        q_table[state][action] = old_q + alpha * (reward + gamma * future_q - old_q)

        if not alive:
            done = True

print("Treinamento finalizado! Q-table pronta.")
print("Q-Table final:")
print(q_table)