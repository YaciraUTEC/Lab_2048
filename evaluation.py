import numpy as np
from game_2048 import Game2048

def evaluate_agent_scalar(agent, seeds, size=4, max_steps=5000):
    scores = []
    max_tiles = []
    steps_list = []

    for sd in seeds:
        game = Game2048(size=size, seed=sd)
        score = 0
        steps = 0

        while True:
            legal = game.legal_actions()
            if not legal:
                break

            action = agent.act(game.board.copy(), legal)
            result = game.step(action)

            if result.info.get("moved", False):
                score += int(result.reward)

            steps += 1
            if result.done or steps >= max_steps:
                break

        scores.append(score)
        max_tiles.append(int(game.board.max()))
        steps_list.append(steps)

    scores = np.array(scores, dtype=np.float64)
    max_tiles = np.array(max_tiles, dtype=np.float64)
    steps_list = np.array(steps_list, dtype=np.float64)

    # Logs
    L = np.log1p(scores)                         # log(1 + score)
    T = np.log2(np.maximum(max_tiles, 1.0))      # log2(max_tile), safe
    K = np.log1p(steps_list)                     # log(1 + steps)

    L_mean = float(L.mean())
    L_med = float(np.median(L))
    T_mean = float(T.mean())
    K_mean = float(K.mean())

    final_score = 1000.0 * L_mean + 30.0 * T_mean + 10.0 * L_med - 2.0 * K_mean

    return {
        "final_score": float(final_score),
        "mean_log_score": L_mean,
        "median_log_score": L_med,
        "mean_log2_max_tile": T_mean,
        "mean_log_steps": K_mean,
        "episodes": int(len(seeds)),
    }


def evaluate_agent_scalar_render(
    agent,
    seeds,
    size=4,
    max_steps=5000,
    step_delay=0.03,
):

    import matplotlib.pyplot as plt
    from viz_2048 import Renderer2048

    plt.ion()
    renderer = Renderer2048.create(size=size, window_title="2048 (evaluation)")

    scores = []
    max_tiles = []
    steps_list = []

    total_eps = len(seeds)

    for ep_idx, sd in enumerate(seeds, start=1):
        if not plt.fignum_exists(renderer.fig.number):
            break

        game = Game2048(size=size, seed=sd)
        score = 0
        steps = 0

        renderer.draw(game.board, score=score, status=f"Seed {sd} ({ep_idx}/{total_eps})")
        plt.pause(0.001)

        while True:
            legal = game.legal_actions()
            if not legal:
                break

            action = agent.act(game.board.copy(), legal)
            result = game.step(action)

            if result.info.get("moved", False):
                score += int(result.reward)

            steps += 1

            if not plt.fignum_exists(renderer.fig.number):
                break

            renderer.draw(
                result.obs,
                score=score,
                status=f"Seed {sd} ({ep_idx}/{total_eps}) | step {steps}",
            )
            plt.pause(step_delay)

            if result.done or steps >= max_steps:
                break

        scores.append(score)
        max_tiles.append(int(game.board.max()))
        steps_list.append(steps)

        if not plt.fignum_exists(renderer.fig.number):
            break

        renderer.draw(game.board, score=score, status=f"Seed {sd} done. max_tile={max_tiles[-1]}")
        plt.pause(0.15)

    plt.ioff()
    if plt.fignum_exists(renderer.fig.number):
        plt.show(block=False)

    scores = np.array(scores, dtype=np.float64)
    max_tiles = np.array(max_tiles, dtype=np.float64)
    steps_list = np.array(steps_list, dtype=np.float64)

    if len(scores) == 0:
        return {
            "final_score": float("nan"),
            "mean_log_score": float("nan"),
            "median_log_score": float("nan"),
            "mean_log2_max_tile": float("nan"),
            "mean_log_steps": float("nan"),
            "episodes": 0,
        }

    # Logs
    L = np.log1p(scores)  # log(1 + score)
    T = np.log2(np.maximum(max_tiles, 1.0))  # log2(max_tile), safe
    K = np.log1p(steps_list)  # log(1 + steps)

    L_mean = float(L.mean())
    L_med = float(np.median(L))
    T_mean = float(T.mean())
    K_mean = float(K.mean())

    final_score = 1000.0 * L_mean + 30.0 * T_mean + 10.0 * L_med - 2.0 * K_mean

    return {
        "final_score": float(final_score),
        "mean_log_score": L_mean,
        "median_log_score": L_med,
        "mean_log2_max_tile": T_mean,
        "mean_log_steps": K_mean,
        "episodes": int(len(scores)),
    }
