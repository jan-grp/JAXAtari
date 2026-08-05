import argparse
import os
import sys

# JAX platform must be chosen before `import jax`.
if "--cpu" in sys.argv:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.random as jrandom
import numpy as np

from jaxatari.environment import JAXAtariAction as Action
from jaxatari.games.jax_icehockey import IceHockeyConstants, JaxIceHockey

UPSCALE = 4


def to_uint8_image(raster) -> np.ndarray:
    """Convert a (H, W, C) JAX raster into an (H, W, 3) uint8 numpy array."""
    img = np.asarray(raster)
    if img.dtype != np.uint8:
        # Floats are assumed to be in [0, 1].
        img = np.clip(img * 255.0 if img.max() <= 1.0 else img, 0, 255).astype(np.uint8)
    if img.ndim == 2:  # grayscale
        img = np.stack([img] * 3, axis=-1)
    if img.shape[-1] == 4:  # drop alpha
        img = img[..., :3]
    return img


def make_env(debug: bool, mods: list = None):
    if mods:
        import jaxatari

        return jaxatari.core.make("icehockey", mods=mods)
    return JaxIceHockey(IceHockeyConstants(DEBUG_RENDER=debug))


def snapshot(out_path: str, scale: int, debug: bool, mods: list = None) -> None:
    env = make_env(debug, mods)
    _obs, state = env.reset(jrandom.PRNGKey(0))
    raster = env.render(state)
    img = to_uint8_image(raster)

    print(
        f"render output: shape={np.asarray(raster).shape} dtype={np.asarray(raster).dtype}"
    )
    print(
        f"value range: min={img.min()} max={img.max()} (all-black means nothing was drawn)"
    )

    if scale > 1:
        img = np.kron(img, np.ones((scale, scale, 1), dtype=np.uint8))

    try:
        from PIL import Image

        Image.fromarray(img).save(out_path)
        print(f"wrote {out_path} ({img.shape[1]}x{img.shape[0]})")
    except ImportError:
        npy_path = os.path.splitext(out_path)[0] + ".npy"
        np.save(npy_path, img)
        print(f"Pillow not installed; saved raw array to {npy_path} instead.")


def get_action(pygame, keys, up_key, down_key, left_key, right_key, fire_key):
    up = keys[up_key]
    down = keys[down_key]
    left = keys[left_key]
    right = keys[right_key]
    fire = keys[fire_key]

    if up and right:
        return Action.UPRIGHTFIRE if fire else Action.UPRIGHT
    if up and left:
        return Action.UPLEFTFIRE if fire else Action.UPLEFT
    if down and right:
        return Action.DOWNRIGHTFIRE if fire else Action.DOWNRIGHT
    if down and left:
        return Action.DOWNLEFTFIRE if fire else Action.DOWNLEFT
    if up:
        return Action.UPFIRE if fire else Action.UP
    if down:
        return Action.DOWNFIRE if fire else Action.DOWN
    if left:
        return Action.LEFTFIRE if fire else Action.LEFT
    if right:
        return Action.RIGHTFIRE if fire else Action.RIGHT
    if fire:
        return Action.FIRE
    return Action.NOOP


def play(scale: int, keyboard_enemy: bool, debug: bool, mods: list = None) -> None:
    import pygame

    env = make_env(debug, mods)
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)
    render_fn = jax.jit(env.render)

    _obs, state = reset_fn(jrandom.PRNGKey(0))
    h, w = env.consts.HEIGHT, env.consts.WIDTH

    pygame.init()
    screen = pygame.display.set_mode((w * scale, h * scale))
    caption = "jax_icehockey render test - arrows/space player, R reset"
    if keyboard_enemy:
        caption += ", WASD/left-shift enemy"
    pygame.display.set_caption(caption)
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_r:
                    _obs, state = reset_fn(jrandom.PRNGKey(0))

        keys = pygame.key.get_pressed()
        action = get_action(
            pygame,
            keys,
            pygame.K_UP,
            pygame.K_DOWN,
            pygame.K_LEFT,
            pygame.K_RIGHT,
            pygame.K_SPACE,
        )
        step_action = action
        if keyboard_enemy:
            enemy_action = get_action(
                pygame,
                keys,
                pygame.K_w,
                pygame.K_s,
                pygame.K_a,
                pygame.K_d,
                pygame.K_LSHIFT,
            )
            step_action = (action, enemy_action)

        step_out = step_fn(state, step_action)
        state = step_out[1]  # (obs, state, reward, done, info)

        img = to_uint8_image(render_fn(state))
        # pygame surfaces are (W, H), so transpose axes 0 and 1.
        surf = pygame.surfarray.make_surface(np.transpose(img, (1, 0, 2)))
        surf = pygame.transform.scale(surf, (w * scale, h * scale))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--play", action="store_true", help="open an interactive pygame window"
    )
    parser.add_argument(
        "--out", default="icehockey_frame.png", help="snapshot output path"
    )
    parser.add_argument("--scale", type=int, default=UPSCALE, help="upscale factor")
    parser.add_argument("--cpu", action="store_true", help="force JAX onto CPU")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="draw character position pixels and puck pickup regions",
    )
    parser.add_argument(
        "--keyboard-enemy",
        action="store_true",
        help="control the enemy with WASD and left shift instead of the NPC controller",
    )
    parser.add_argument(
        "--mods",
        nargs="*",
        default=None,
        help="mod keys to apply via jaxatari.core.make, e.g. --mods change_border_shape",
    )
    args = parser.parse_args()

    if args.play:
        play(args.scale, args.keyboard_enemy, args.debug, args.mods)
    else:
        snapshot(args.out, args.scale, args.debug, args.mods)


if __name__ == "__main__":
    main()
