import os
import numpy as np
import jaxatari.games
from jaxatari.games.jax_icehockey import IceHockeyConstants
from jaxatari.modification import JaxAtariInternalModPlugin


def _make_narrowed_goal_background(new_x0: int, new_x1: int) -> np.ndarray:
    """Load the icehockey background and close the goal mouths down to [new_x0, new_x1).

    The rink is baked into the background sprite: each goal is a black gap in
    the grey board band (rows PLAYER_GOAL_Y..+GOAL_HEIGHT at the top,
    ENEMY_GOAL_Y-GOAL_HEIGHT+1..ENEMY_GOAL_Y at the bottom, columns
    GOAL_X0..GOAL_X1). The now-covered columns are filled with board pixels.
    """
    c = IceHockeyConstants()
    sprite_path = os.path.join(
        os.path.dirname(jaxatari.games.__file__),
        "sprites",
        "icehockey",
        "background.npy",
    )
    bg = np.load(sprite_path).copy()
    board = np.array([192, 192, 192, 255], dtype=np.uint8)
    top_rows = slice(c.PLAYER_GOAL_Y, c.PLAYER_GOAL_Y + c.GOAL_HEIGHT)
    bottom_rows = slice(c.ENEMY_GOAL_Y - c.GOAL_HEIGHT + 1, c.ENEMY_GOAL_Y + 1)
    for rows in (top_rows, bottom_rows):
        bg[rows, c.GOAL_X0:new_x0] = board
        bg[rows, new_x1:c.GOAL_X1] = board
    return bg


# --- 1. Individual Mod Plugins ---
class NoAttackingZonesMod(JaxAtariInternalModPlugin):
    """Removes the attacking-zone restrictions from the rink.

    In the base game ATTACKING_ZONE_OFFSET_Y carves a restricted band in front
    of each goal: a skater is kept out of its own defensive zone and a goalie
    out of the opponent's far zone (movement bounds in _characters_step), and
    the same constant drives the zone-based active-character switching in
    _resolve_active_characters. Overriding it to 0 collapses those zones, so
    all four characters may skate the full rink and the active character falls
    back to the closest-to-puck rule outside the goal areas.
    """

    constants_overrides = {
        "ATTACKING_ZONE_OFFSET_Y": 0,
    }


class DisableTacklingMod(JaxAtariInternalModPlugin):
    """Disables body-checks: no character can ever be knocked down.

    A tackle only lands in _tackle_step when a per-swing random roll is below
    TACKLE_SUCCESS_PROB; with the probability forced to 0.0 the roll
    (uniform in [0, 1)) can never succeed. The FIRE swing itself is untouched,
    so shooting the puck and the swing animation still work as in the base
    game -- opponents just never go down.
    """

    constants_overrides = {
        "TACKLE_SUCCESS_PROB": 0.0,
    }


class DecreasedGoalSizeMod(JaxAtariInternalModPlugin):
    """Halves the width of both goals (mouth 64..96 -> 72..88, centred).

    GOAL_X0/GOAL_X1 drive goal detection in _goal_and_reset_step and the
    goalie-protection band in _goalie_protected, so shots outside the narrowed
    mouth now bounce off the boards instead of scoring. The background asset
    is rebuilt with the covered goal columns filled in as boards so the visuals
    match the new geometry.
    """

    _NEW_GOAL_X0 = 72
    _NEW_GOAL_X1 = 88

    constants_overrides = {
        "GOAL_X0": _NEW_GOAL_X0,
        "GOAL_X1": _NEW_GOAL_X1,
    }
    asset_overrides = {
        "background": {
            "name": "background",
            "type": "background",
            "data": _make_narrowed_goal_background(_NEW_GOAL_X0, _NEW_GOAL_X1),
        }
    }
