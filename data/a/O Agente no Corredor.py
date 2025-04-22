import numpy as np
import random

# Parâmetros
num_states = 5
num_actions = 2  # 0 = esquerda, 1 = direita
q_table = np.zeros((num_states, num_actions))

alpha = 0.1       # taxa de aprendizado
gamma = 0.9       # fator de desconto
epsilon = 0.2     # exploração vs. exploração
num_episodes = 100

# Função de recompensa
def get_reward(state):
    return 10 if state == 4 else 0

# Loop de treinamento
for episode in range(num_episodes):
    state = 0
    done = False

    while not done:
        # Escolhe ação com ε-greedy
        if random.uniform(0, 1) < epsilon:
            action = random.randint(0, 1)  # Exploração
        else:
            action = np.argmax(q_table[state])  # Exploração

        # Aplica ação
        if action == 0:  # Esquerda
            next_state = max(0, state - 1)
        else:  # Direita
            next_state = min(num_states - 1, state + 1)

        reward = get_reward(next_state)

        # Atualiza Q-table
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        new_value = old_value + alpha * (reward + gamma * next_max - old_value)
        q_table[state, action] = new_value

        # Move para próximo estado
        state = next_state

        if state == 4:
            done = True

# Mostra a Q-table final
print("Q-Table final:")
print(q_table)
