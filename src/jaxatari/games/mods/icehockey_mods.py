import os
from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.icehockey.icehockey_mod_plugins import (
    ChangeBorderShapeMod,
    EnemySpeedUpMod,
    MovingGoalsMod,
    PlayerSlidingMod,
)


class IceHockeyEnvMod(JaxAtariModController):
    """
    Game-specific Mod Controller for IceHockey.
    It simply inherits all logic from JaxAtariModController and defines the ICEHOCKEY_MOD_REGISTRY.
    """

    REGISTRY = {
        "change_border_shape": ChangeBorderShapeMod,
        "moving_goals": MovingGoalsMod,
        "player_sliding": PlayerSlidingMod,
        "enemy_speedup_on_goal": EnemySpeedUpMod,
    }

    _mod_sprite_dir = os.path.join(os.path.dirname(__file__), "icehockey", "sprites")

    def __init__(self,
                 env,
                 mods_config: list = [],
                 allow_conflicts: bool = False
                 ):

        super().__init__(
            env=env,
            mods_config=mods_config,
            allow_conflicts=allow_conflicts,
            registry=self.REGISTRY
        )
