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

from safety_gymnasium.assets.geoms import Zones
from safety_gymnasium.tasks.ltl.ltl_base_task import LtlBaseTask


class LtlLevel0(LtlBaseTask):
    """No zones."""

    def __init__(self, config) -> None:
        super().__init__(config=config, zone_size=0.4)
        self._add_geoms(Zones(color='green', size=self.zone_size, num=2))
