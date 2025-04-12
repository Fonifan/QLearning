import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def create_vid(q_history, env, filename='heatmap_video.avi'):
    first_Q = q_history[0]
    xs = [coord[0] for coord in first_Q.keys()]
    ys = [coord[1] for coord in first_Q.keys()]
    grid_x = max(xs) + 1
    grid_y = max(ys) + 1
    frame_size = (640, 480)

    fig, ax = plt.subplots()
    canvas = FigureCanvas(fig)
    
    Q_max = np.zeros((grid_x, grid_y))
    best_actions = np.zeros((grid_x, grid_y), dtype=int)
    im = ax.imshow(Q_max, cmap="hot", interpolation="nearest", origin="upper")
    cbar = fig.colorbar(im, ax=ax, label="Max Q Value")
    texts = [[ax.text(j, i, "", color="blue", ha="center", va="center", fontsize=12)
              for j in range(grid_y)] for i in range(grid_x)]
    ax.set_title("")
    ax.set_xlabel("Y")
    ax.set_ylabel("X")
    
    out = cv2.VideoWriter(
        filename,
        cv2.VideoWriter_fourcc(*'XVID'),
        10,
        frame_size
    )

    for idx, Q in enumerate(q_history):
        print("Generating frame for episode", idx)

        Q_max.fill(0)
        best_actions.fill(0)
        for (i, j), q_values in Q.items():
            Q_max[i, j] = max(q_values)
            best_actions[i, j] = np.argmax(q_values)
        im.set_data(Q_max)
        im.set_clim(np.min(Q_max), np.max(Q_max))

        for i in range(grid_x):
            for j in range(grid_y):
                texts[i][j].set_text(str(env.pretty_print_action(best_actions[i, j])))

        ax.set_title(f"Q Heatmap at Episode {idx}")
        
        canvas.draw()
        s, (w, h) = canvas.print_to_buffer()
        img = np.frombuffer(s, np.uint8).reshape((h, w, 4))
        frame = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2BGR)
        frame = cv2.resize(frame, frame_size)
        out.write(frame)

    out.release()
    plt.close(fig)
    print(f"Video saved as {filename}")