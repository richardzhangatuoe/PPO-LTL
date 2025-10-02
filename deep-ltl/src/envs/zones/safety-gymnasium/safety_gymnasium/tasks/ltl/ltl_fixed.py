# Copyright 2025 Mathias Jackermeier. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""LTL level 0."""
import numpy as np
from safety_gymnasium.assets.geoms import Zones
from safety_gymnasium.tasks.ltl.ltl_base_task import LtlBaseTask

class LtlFixed(LtlBaseTask):
    """Two green, two yellow, two red, and two magenta zones."""

    def __init__(self, config) -> None:
        super().__init__(config=config, zone_size=0.4)

        # Optimality
        # self._add_geoms(Zones(color='green', size=self.zone_size, num=1, locations=[(-1.4, -1.9)], keepout=0))
        # self._add_geoms(Zones(color='yellow', size=self.zone_size, num=1, locations=[(1.4, 2.3)]))
        # self._add_geoms(Zones(color='blue', size=self.zone_size, num=2, locations=[(-0.8, -1), (1, 1)], keepout=0))
        # self._add_geoms(Zones(color='magenta', size=self.zone_size, num=1, locations=[(-1.7, 1.2)], keepout=0))
        # self._set_agent_location((0.2, 0.2))

        # Safety
        self._add_geoms(Zones(color='green', size=self.zone_size, num=1, locations=[(1.2, -1.9)], keepout=0))
        self._add_geoms(Zones(color='yellow', size=self.zone_size, num=1, locations=[(1.1, 2.1)]))
        self._add_geoms(Zones(color='blue', size=self.zone_size, num=3, locations=[(2, -1), (.6, -1.05), (.1, -2.3)], keepout=0))
        self._add_geoms(Zones(color='magenta', size=self.zone_size, num=1, locations=[(1.8, 0.4)], keepout=0))
        self._set_agent_location((-1.2, -.6))

        # self._set_agent_rotation(np.pi / 2)
