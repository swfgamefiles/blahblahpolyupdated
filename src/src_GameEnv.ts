// GameEnv.ts
// This module creates a simple simulation environment for RL.
// The state is represented by {step, score}.
// There are two discrete actions:
//   0 = no action
//   1 = action that tries to boost the score.
// The reward is generated based on the chosen action.
// Episodes last for a fixed number of steps (default 20).

export interface State {
  step: number;
  score: number;
}

export class GameEnv {
  totalSteps: number;
  currentState: State;

  constructor(totalSteps = 20) {
    this.totalSteps = totalSteps;
    this.currentState = { step: 0, score: 0 };
  }

  reset(): State {
    this.currentState = { step: 0, score: 0 };
    return this.currentState;
  }

  // Takes an action and returns the next state, the reward, and whether the episode is done.
  // Action 0: no boost, reward is low.
  // Action 1: boost, reward is higher.
  step(action: number): { nextState: State; reward: number; done: boolean } {
    const { step, score } = this.currentState;
    let reward = 0;
    if (action === 1) {
      // Boost action: higher, but variable reward.
      reward = Math.floor(Math.random() * 20) + 5; // rewards between 5 and 24
    } else {
      // No boost: lower reward.
      reward = Math.floor(Math.random() * 5); // reward between 0 and 4
    }
    const nextStep = step + 1;
    const nextScore = score + reward;

    this.currentState = { step: nextStep, score: nextScore };
    const done = nextStep >= this.totalSteps;
    return { nextState: this.currentState, reward, done };
  }
}