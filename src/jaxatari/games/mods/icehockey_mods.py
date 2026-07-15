from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.icehockey.icehockey_mod_plugins import (
    NoAttackingZonesMod,
    DecreasedGoalSizeMod,
    DisableTacklingMod,
)


class IceHockeyEnvMod(JaxAtariModController):
    """
    Game-specific Mod Controller for Ice Hockey.
    It simply inherits all logic from JaxAtariModController and defines the REGISTRY.
    """

    REGISTRY = {
        "no_attacking_zones": NoAttackingZonesMod,
        "decreased_goal_size": DecreasedGoalSizeMod,
        "disable_tackling": DisableTacklingMod,
    }

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
