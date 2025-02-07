// RLAgent.ts
// This module implements a basic Q-learning agent.
// It uses an epsilon-greedy policy to choose actions and updates its Q‑table based on observed rewards.
// The state is represented by the current step (as extracted from the environment's state).

import { State } from "./GameEnv.ts";

export class RLAgent {
  actions: number[]; // List of discrete actions
  alpha: number; // Learning rate
  gamma: number; // Discount factor
  epsilon: number; // Exploration rate
  qTable: Map<string, number>;

  constructor(actions: number[], alpha = 0.1, gamma = 0.99, epsilon = 1.0) {
    this.actions = actions;
    this.alpha = alpha;
    this.gamma = gamma;
    this.epsilon = epsilon;
    this.qTable = new Map<string, number>();
  }

  // Helper: converts a state to a string key.
  // Here we use "step" as the state representation.
  stateKey(state: State): string {
    return `${state.step}`;
  }

  // Retrieves the Q-value for a given state/action pair.
  getQ(state: State, action: number): number {
    const key = this.stateKey(state) + `|${action}`;
    if (!this.qTable.has(key)) {
      this.qTable.set(key, 0);
    }
    return this.qTable.get(key)!;
  }

  // Sets the Q-value for a given state/action pair.
  setQ(state: State, action: number, value: number): void {
    const key = this.stateKey(state) + `|${action}`;
    this.qTable.set(key, value);
  }

  // Chooses an action using an epsilon-greedy approach.
  chooseAction(state: State): number {
    if (Math.random() < this.epsilon) {
      // Explore: choose a random action.
      return this.actions[Math.floor(Math.random() * this.actions.length)];
    } else {
      // Exploit: choose the action with the highest Q-value.
      let bestAction = this.actions[0];
      let bestQ = this.getQ(state, bestAction);
      for (let action of this.actions) {
        const q = this.getQ(state, action);
        if (q > bestQ) {
          bestQ = q;
          bestAction = action;
        }
      }
      return bestAction;
    }
  }

  // Updates the Q-value using the standard Q-learning update rule.
  update(state: State, action: number, reward: number, nextState: State, done: boolean): void {
    const currentQ = this.getQ(state, action);
    let maxNextQ = 0;
    if (!done) {
      maxNextQ = Math.max(...this.actions.map(a => this.getQ(nextState, a)));
    }
    const newQ = currentQ + this.alpha * (reward + this.gamma * maxNextQ - currentQ);
    this.setQ(state, action, newQ);
  }

  // Applies decay to the exploration rate after each episode.
  decayEpsilon(decayRate: number): void {
    this.epsilon = this.epsilon * decayRate;
  }
}