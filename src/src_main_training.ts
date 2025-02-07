// main_training.ts
// This file runs the training loop for the Q-learning agent interacting with the GameEnv.
// The training loop is executed entirely in the browser, and progress is printed to the page.

import { GameEnv } from "./GameEnv.ts";
import { RLAgent } from "./RLAgent.ts";

// Obtain a reference to the output element in the DOM.
const outputElement = document.getElementById("output");

/**
 * Logs a message to the output element.
 */
function logMessage(message: string): void {
  if (outputElement) {
    outputElement.textContent += message + "\n";
  }
}

// A delay helper function.
function delay(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function trainAgent(episodes: number) {
  const env = new GameEnv(20);
  const agent = new RLAgent([0, 1], 0.1, 0.99, 1.0);
  const decayRate = 0.99;

  for (let ep = 1; ep <= episodes; ep++) {
    let state = env.reset();
    let done = false;
    let totalReward = 0;

    while (!done) {
      const action = agent.chooseAction(state);
      const { nextState, reward, done: episodeDone } = env.step(action);
      agent.update(state, action, reward, nextState, episodeDone);
      state = nextState;
      totalReward += reward;
      done = episodeDone;
      // Slow down loop for visualization purposes (optional)
      await delay(10);
    }

    agent.decayEpsilon(decayRate);
    logMessage(`Episode ${ep} finished with total reward: ${totalReward} and epsilon: ${agent.epsilon.toFixed(3)}`);
  }

  logMessage("Training completed. Learned Q-Table:");
  // Log out each Q-value entry.
  agent.qTable.forEach((value, key) => {
    logMessage(`${key}: ${value.toFixed(3)}`);
  });
}

// Start training when the window loads.
window.addEventListener("load", async () => {
  logMessage("Starting RL agent training...");
  await trainAgent(100);
  logMessage("RL training done.");
});